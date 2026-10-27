"""Finishing pass on an exported Pixal3D GLB: loose fragments removed, texture colour measured against the photos
(and, with --grade, corrected), re-exported without a second lossy texture encode.

    python /tools/glb_finish.py <in.glb> <gen.pt> <input_dir> <out.glb> [--keep-frac 0.01] [--grade]

Fragments: connected components are found on a copy welded across UV seams (positions only); components with
fewer than keep-frac of the faces are dropped from the textured mesh, UVs and all.
Colour: through each camera of <input_dir>/transforms.json the GLB's flat albedo (tools/glb_render.py) is compared
with the photo on pixels well inside both silhouettes: per-channel median of photo / albedo. Under white light
that ratio is shading only and the same in every channel; a spread between channels is a cast in the texture.
--grade multiplies the base-colour texture by per-channel gains that remove that spread (the pooled median ratio
over all views, normalised to unit luminance so the texture's level, which already matches the photos on average,
is left alone: no assumption about the photos' lighting level is made), and logs them.
Textures are written as PNG (lossless) so the WebP the exporter wrote is not encoded twice. The written file is
reloaded and checked.
"""
import json
import os
import sys

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image

sys.path.insert(0, "/work")
sys.path.insert(0, "/tools")
import views  # noqa: E402
from frames import glb_to_raw, world_from_raw  # noqa: E402
from glb_render import render_glb, uv_orientation  # noqa: E402


def to_projection(g, pt):
    d = torch.load(pt, map_location="cpu")
    v = d["vertices"].double().numpy()
    coords = d["coords"].double().numpy()
    if coords.shape[1] == 4:
        coords = coords[:, 1:]
    T = world_from_raw(v, coords, d["res"])[1] @ glb_to_raw(np.asarray(g.vertices, np.float64), v)
    assert abs(np.linalg.det(T) - 1) < 1e-9
    return T


def gpu_mesh(g, T):
    P = torch.tensor(np.asarray(g.vertices) @ T.T, dtype=torch.float32, device="cuda")
    F = torch.tensor(np.asarray(g.faces), dtype=torch.long, device="cuda")
    fn = torch.linalg.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    N = torch.zeros_like(P).index_add_(0, F.reshape(-1), fn.repeat_interleave(3, 0))
    N = N / N.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    UV = torch.tensor(np.asarray(g.visual.uv), dtype=torch.float32, device="cuda")
    return P, F, N, UV


def colour_stats(g, T, inp, flip_v):
    meta = json.load(open(os.path.join(inp, "transforms.json")))
    views.project.__defaults__ = (float(meta["camera_angle_x"]),)
    tex = np.asarray(g.visual.material.baseColorTexture.convert("RGB"))
    P, F, N, UV = gpu_mesh(g, T)
    ratios = []
    out = {}
    for fr in meta["frames"]:
        _, flat, mm = render_glb(P, F, N, UV, tex, fr["transform_matrix"], 1024, flip_v)
        x = np.asarray(Image.open(os.path.join(inp, fr["file_path"])).convert("RGBA").resize((1024, 1024), Image.Resampling.LANCZOS)).astype(np.float32) / 255
        inside = cv2.erode(((x[..., 3] > 0.99) & mm).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        inside &= flat.min(-1) > 0.05
        r = x[..., :3][inside] / flat[inside]
        ratios.append(r)
        med = np.median(r, 0)
        out[fr["name"]] = {"px": int(inside.sum()), "ratio_median": med.tolist(), "ratio_median_norm": (med / med.mean()).tolist(),
                           "albedo_median": np.median(flat[inside], 0).tolist(), "photo_median": np.median(x[..., :3][inside], 0).tolist()}
        print(f"[colour {fr['name']}] {inside.sum()} px: photo/albedo median {med.round(3)} (normalised {(med / med.mean()).round(3)}); "
              f"albedo median {np.median(flat[inside], 0).round(3)}, photo median {np.median(x[..., :3][inside], 0).round(3)}", flush=True)
    r = np.concatenate(ratios)
    med = np.median(r, 0)
    lum = r @ np.array([0.2126, 0.7152, 0.0722])
    lit = np.percentile(lum, 90)
    out["all"] = {"ratio_median": med.tolist(), "ratio_lum_p90": float(lit)}
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    keep_frac = float(sys.argv[sys.argv.index("--keep-frac") + 1]) if "--keep-frac" in sys.argv else 0.01
    if "--keep-frac" in sys.argv:
        args.remove(sys.argv[sys.argv.index("--keep-frac") + 1])
    grade = "--grade" in sys.argv
    src, pt, inp, dst = args
    info = {"source": src}

    scene = trimesh.load(src, force="scene")
    geoms = scene.dump()
    assert len(geoms) == 1, f"{len(geoms)} geometries"
    g = geoms[0]
    nf0, nv0 = len(g.faces), len(g.vertices)

    welded = trimesh.Trimesh(vertices=np.asarray(g.vertices), faces=np.asarray(g.faces), process=False)
    welded.merge_vertices(merge_tex=True, merge_norm=True)
    assert len(welded.faces) == nf0
    labels = trimesh.graph.connected_component_labels(welded.face_adjacency, node_count=nf0)
    sizes = np.bincount(labels)
    order = np.argsort(sizes)[::-1]
    keep_labels = [l for l in order if sizes[l] >= keep_frac * nf0]
    dropped = [(int(l), int(sizes[l])) for l in order if sizes[l] < keep_frac * nf0]
    print(f"[fragments] {len(sizes)} components; kept {len(keep_labels)} ({', '.join(str(int(sizes[l])) for l in keep_labels)} faces), "
          f"dropped {len(dropped)} with {sum(s for _, s in dropped)} faces (largest dropped {dropped[0][1] if dropped else 0})", flush=True)
    for l, s in dropped[:5]:
        c = welded.vertices[np.unique(welded.faces[labels == l])]
        print(f"[fragments]   dropped component: {s} faces, centre {c.mean(0).round(4)}, extent {np.ptp(c, 0).round(4)}", flush=True)
    info["fragments"] = {"components": int(len(sizes)), "kept_faces": [int(sizes[l]) for l in keep_labels], "dropped": dropped}
    normals = np.asarray(g.vertex_normals).copy()  # the exporter's own (uvbake/cumesh); editing faces would drop them
    mat = g.visual.material
    src_tex = {k: np.asarray(getattr(mat, k)).copy() for k in ("baseColorTexture", "metallicRoughnessTexture") if getattr(mat, k, None) is not None}
    keep_f = np.isin(labels, keep_labels)
    used = np.zeros(nv0, bool)
    used[np.asarray(g.faces)[keep_f].ravel()] = True
    g.update_faces(keep_f)
    g.remove_unreferenced_vertices()
    assert len(g.vertices) == used.sum()
    g.vertex_normals = normals[used]
    print(f"[fragments] {nv0} -> {len(g.vertices)} vertices, {nf0} -> {len(g.faces)} faces", flush=True)

    T = to_projection(g, pt)
    meta = json.load(open(os.path.join(inp, "transforms.json")))
    views.project.__defaults__ = (float(meta["camera_angle_x"]),)
    flip_v, corr = uv_orientation(*gpu_mesh(g, T), np.asarray(g.visual.material.baseColorTexture.convert("RGB")), pt,
                                  meta["frames"][0]["transform_matrix"])
    print(f"[uv] albedo correlation with the raw generation: as stored {corr[False]:.3f}, V flipped {corr[True]:.3f}", flush=True)
    info["uv_v_flipped"] = flip_v
    info["colour_before"] = colour_stats(g, T, inp, flip_v)

    if grade:
        med = np.array(info["colour_before"]["all"]["ratio_median"])
        gain = med / (med @ np.array([0.2126, 0.7152, 0.0722]))  # cast only, luminance kept
        rgba = np.asarray(g.visual.material.baseColorTexture).copy()  # RGBA: uvbake packs alpha in; keep it as is
        tex = rgba[..., :3].astype(np.float32)
        clipped = float((tex * gain > 255).any(-1).mean())
        rgba[..., :3] = np.clip(np.rint(tex * gain), 0, 255).astype(np.uint8)
        g.visual.material.baseColorTexture = Image.fromarray(rgba, g.visual.material.baseColorTexture.mode)
        info["grade"] = {"gain": gain.tolist(), "texels_clipped_frac": clipped}
        print(f"[grade] per-channel gain {gain.round(4)} (cast removed, luminance kept); texels clipped {clipped:.4f}", flush=True)
        info["colour_after"] = colour_stats(g, T, inp, flip_v)

    g.export(dst, extension_webp=False, include_normals=True)
    back = trimesh.load(dst, force="scene").dump()
    assert len(back) == 1 and len(back[0].faces) == len(g.faces) and len(back[0].vertices) == len(g.vertices)
    mr = back[0].visual.material
    for k, before in src_tex.items():
        now = np.asarray(getattr(g.visual.material, k))
        if not (grade and k == "baseColorTexture"):
            assert np.array_equal(now, before), f"{k} changed before export"
        assert np.array_equal(np.asarray(getattr(mr, k)), now), f"{k} changed in the round trip"
    if grade:
        assert np.array_equal(np.asarray(mr.baseColorTexture)[..., 3], src_tex["baseColorTexture"][..., 3]), "alpha changed"
    dn = np.abs(np.asarray(back[0].vertex_normals) - normals[used]).max()
    assert dn < 1e-3, f"normals changed by {dn}"
    info["normals_max_change"] = float(dn)
    info["written"] = {"path": dst, "bytes": os.path.getsize(dst), "vertices": len(back[0].vertices), "faces": len(back[0].faces),
                       "textures": {k: (getattr(mr, k).size if getattr(mr, k, None) is not None else None)
                                    for k in ("baseColorTexture", "metallicRoughnessTexture", "normalTexture")}}
    print(f"[written] {dst}: {os.path.getsize(dst) / 1e6:.1f} MB, {len(back[0].vertices)} vertices, {len(back[0].faces)} faces; "
          f"reloaded: textures {info['written']['textures']} bit-identical to what was written, normals within {dn:.1e}", flush=True)
    json.dump(info, open(os.path.splitext(dst)[0] + ".json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
