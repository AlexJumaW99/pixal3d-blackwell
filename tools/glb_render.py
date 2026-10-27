"""Renders of an exported GLB (the deliverable: decimated mesh + baked texture) through a multi-view input's cameras,
and a turntable.

    python /tools/glb_render.py <gen.glb> <gen.pt> <input_dir> <out_dir> [--size 1024]

The GLB's frame is not assumed: it is matched to the raw generation, and the raw generation to the projection frame,
on the geometry (tools/frames.py), so transforms.json's cameras apply exactly. The texture's V orientation is not assumed either (loaders differ on
flipping it): both are rendered and the one whose albedo correlates with the raw generation's own base colour
(sampled as tools/render_check.py does) is kept. Writes <out_dir>/glb_<frame>.png, glb_turntable.jpg and
glb_sheet.jpg (photo | GLB render per frame), and glb.json.
"""
import json
import math
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
from fit_view import camera  # noqa: E402
from frames import glb_to_raw, world_from_raw  # noqa: E402
from render_check import load_gen, render as render_raw  # noqa: E402


def light(nrm, view, cam):
    key = -0.55 * cam["x"] + 0.65 * cam["y"] + 0.52 * cam["z"]
    key = key / key.norm()
    fill = 0.6 * cam["x"] + 0.1 * cam["y"] + 0.8 * cam["z"]
    fill = fill / fill.norm()
    lam = 0.08 + 0.72 * (nrm @ key).clamp(min=0) + 0.22 * (nrm @ fill).clamp(min=0)
    h = (key + view) / (key + view).norm(dim=-1, keepdim=True)
    return lam, 0.35 * (nrm * h).sum(-1).clamp(min=0) ** 24


def render_glb(P, F, N, UV, tex, c2w, size, flip_v):
    c2w = torch.tensor(c2w, dtype=torch.float32, device="cuda")
    cam = {"eye": c2w[:3, 3], "x": c2w[:3, 0], "y": c2w[:3, 1], "z": c2w[:3, 2]}
    fid, bary, _ = views.rasterize(P, F, cam, size)
    m = fid >= 0
    tri = F[fid[m]]
    b = bary[m].unsqueeze(-1)
    uv = (UV[tri] * b).sum(1).cpu().numpy()
    H, W = tex.shape[:2]
    mx = (uv[:, 0] * W - 0.5).astype(np.float32)
    my = (((1 - uv[:, 1]) if flip_v else uv[:, 1]) * H - 0.5).astype(np.float32)
    n, row = len(mx), 4096  # cv2.remap wants both sides < 32767: lay the lookups out in rows
    pad = (-n) % row
    mx2, my2 = np.pad(mx, (0, pad)).reshape(-1, row), np.pad(my, (0, pad)).reshape(-1, row)
    alb = cv2.remap(tex, mx2, my2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP).reshape(-1, 3)[:n].astype(np.float32) / 255
    nrm = (N[tri] * b).sum(1)
    nrm = nrm / nrm.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    view = cam["eye"] - (P[tri] * b).sum(1)
    view = view / view.norm(dim=-1, keepdim=True)
    nrm = torch.where((nrm * view).sum(-1, keepdim=True) < 0, -nrm, nrm)
    lam, spec = light(nrm, view, cam)
    out = np.zeros((size, size, 3), np.float32)
    mm = m.cpu().numpy()
    out[mm] = np.clip(alb * (0.35 + 0.65 * lam[:, None].cpu().numpy()) + 0.5 * spec[:, None].cpu().numpy(), 0, 1)
    flat = np.zeros((size, size, 3), np.float32)
    flat[mm] = alb
    return out, flat, mm


def uv_orientation(P, F, N, UV, tex, pt, c2w):
    """which V orientation of the texture reproduces the raw generation's own base colour through camera c2w:
    returns (flip_v, {False: corr, True: corr})"""
    world, f_raw, n_raw, col = load_gen(pt)
    _, alb_raw, m_raw = render_raw(world, f_raw, n_raw, col, c2w, 512)
    del world, f_raw, n_raw, col
    torch.cuda.empty_cache()
    corr = {}
    for flip in (False, True):
        _, flat, mm = render_glb(P, F, N, UV, tex, c2w, 512, flip)
        both = mm & m_raw
        corr[flip] = float(np.corrcoef(flat[both].mean(-1), alb_raw[both].mean(-1))[0, 1])
    flip_v = max(corr, key=corr.get)
    assert corr[flip_v] > 0.5, "neither UV orientation reproduces the raw generation's colour"
    return flip_v, corr


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    size = int(sys.argv[sys.argv.index("--size") + 1]) if "--size" in sys.argv else 1024
    if "--size" in sys.argv:
        args.remove(str(size))
    glb, pt, inp, out = args
    os.makedirs(out, exist_ok=True)
    meta = json.load(open(os.path.join(inp, "transforms.json")))
    fov = float(meta["camera_angle_x"])
    views.project.__defaults__ = (fov,)  # views.py binds the Amina run's FOV as project()'s default

    scene = trimesh.load(glb, force="scene")
    geoms = scene.dump()
    assert len(geoms) == 1, f"{len(geoms)} geometries"
    g = geoms[0]
    tex = np.asarray(g.visual.material.baseColorTexture.convert("RGB"))
    uv = np.asarray(g.visual.uv, np.float64)

    # GLB frame -> raw vertices -> projection frame
    d = torch.load(pt, map_location="cpu")
    v = d["vertices"].double().numpy()
    coords = d["coords"].double().numpy()
    if coords.shape[1] == 4:
        coords = coords[:, 1:]
    T = world_from_raw(v, coords, d["res"])[1] @ glb_to_raw(np.asarray(g.vertices, np.float64), v)
    assert abs(np.linalg.det(T) - 1) < 1e-9, "GLB frame and projection frame differ by a mirror"
    Pw = torch.tensor(np.asarray(g.vertices) @ T.T, dtype=torch.float32, device="cuda")
    F = torch.tensor(np.asarray(g.faces), dtype=torch.long, device="cuda")
    fn = torch.linalg.cross(Pw[F[:, 1]] - Pw[F[:, 0]], Pw[F[:, 2]] - Pw[F[:, 0]])
    N = torch.zeros_like(Pw).index_add_(0, F.reshape(-1), fn.repeat_interleave(3, 0))
    N = N / N.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    UV = torch.tensor(uv, dtype=torch.float32, device="cuda")
    print(f"[glb] {len(Pw)} vertices, {len(F)} faces, texture {tex.shape[1]}x{tex.shape[0]}; GLB -> projection frame\n{T.round(6)}", flush=True)

    # V orientation from data: against the raw generation's base colour through frame 0
    flip_v, corr = uv_orientation(Pw, F, N, UV, tex, pt, meta["frames"][0]["transform_matrix"])
    print(f"[uv] albedo correlation with the raw generation: as stored {corr[False]:.3f}, V flipped {corr[True]:.3f} -> {'flipped' if flip_v else 'as stored'}", flush=True)

    rows, info = [], {"uv_v_flipped": flip_v, "uv_corr": {str(k): v_ for k, v_ in corr.items()}, "frames": {}}
    for fr in meta["frames"]:
        lit, _, mm = render_glb(Pw, F, N, UV, tex, fr["transform_matrix"], size, flip_v)
        img = Image.open(os.path.join(inp, fr["file_path"])).convert("RGBA").resize((size, size), Image.Resampling.LANCZOS)
        x = np.asarray(img).astype(np.float32) / 255
        photo = x[..., :3] * x[..., 3:]
        a = x[..., 3] > 0.5
        info["frames"][fr["name"]] = {"silhouette_iou": float((a & mm).sum() / (a | mm).sum())}
        Image.fromarray((lit * 255).round().astype(np.uint8)).save(os.path.join(out, f"glb_{fr['name']}.png"))
        rows.append(np.concatenate([photo, lit], 1))
        print(f"[{fr['name']}] GLB silhouette IoU vs the view's alpha {info['frames'][fr['name']]['silhouette_iou']:.4f}", flush=True)
    sheet = np.concatenate(rows, 0)
    Image.fromarray((cv2.resize(sheet, (sheet.shape[1] // 2, sheet.shape[0] // 2), interpolation=cv2.INTER_AREA) * 255).round().astype(np.uint8)).save(
        os.path.join(out, "glb_sheet.jpg"), quality=92)

    d0 = float(np.linalg.norm(np.array(meta["frames"][0]["transform_matrix"])[:3, 3]))
    tiles = []
    for yaw in range(0, 360, 45):
        lit, _, _ = render_glb(Pw, F, N, UV, tex, camera(yaw, 0, 0, d0 * 1.08, 0, 0), 512, flip_v)
        tiles.append(lit)
    tt = np.concatenate([np.concatenate(tiles[:4], 1), np.concatenate(tiles[4:], 1)], 0)
    Image.fromarray((tt * 255).round().astype(np.uint8)).save(os.path.join(out, "glb_turntable.jpg"), quality=92)
    json.dump(info, open(os.path.join(out, "glb.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
