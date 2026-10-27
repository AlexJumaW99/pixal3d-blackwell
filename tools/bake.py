"""Bake the Amina statue's texture from the source image and the restyled views, by camera projection.

    python /tools/bake.py <mesh.npz> <views_dir> <restyle_dir> <source_rgba.png> <out_dir> [size=4096]

Every texel of the mesh's UV atlas is projected into each view (tools/views.py cameras): the source
photograph from Pixal3D's camera (its square crop of the source, CROP below) and each restyled
render (mv/restyle.py). A view counts where the texel is the nearest surface there (its depth map)
and clear of the view's outline (masks eroded: the grey background bleeds into the edge), weighted
by how squarely it faces that camera; the source counts SOURCE_WEIGHT times as much. The restyled
views are first given the source's tones on the statue (histogram matching), so they blend into it.
Writes <out>/albedo.png (grey, the image as photographed: the lighting is in it) and coverage stats.
"""
import json
import math
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image

sys.path.insert(0, "/tools")
from uvbake import uv_rasterize  # noqa: E402
from views import camera, project, FOV  # noqa: E402

DEV = os.environ.get("BAKE_DEVICE", "cuda")
CROP = (-108, -42, 1354)  # Pixal3D's square crop of the 1024x1280 source: x, y, size (out/sv1024)
SOURCE_WEIGHT = 30.0
FACING_POWER = 3.0
EDGE_ERODE = {"source": 3, "restyled": 2}  # px at 1024: the source's soft matte edge; the restyled views are pixel-locked to the render


def grey(a):
    a = a.astype(np.float32) / 255
    return 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]


def match_hist(src_vals, ref_vals):
    """a monotone curve taking src's distribution to ref's (both 1-D, 0..1): returns a function"""
    qs = np.linspace(0, 1, 257)
    s = np.quantile(src_vals, qs)
    r = np.quantile(ref_vals, qs)
    s = np.maximum.accumulate(s + np.arange(len(s)) * 1e-7)
    return lambda x: np.interp(x, s, r)


def mask_of_restyled(img):
    bg = np.median(np.concatenate([img[:20].reshape(-1, 3), img[-20:].reshape(-1, 3), img[:, :20].reshape(-1, 3), img[:, -20:].reshape(-1, 3)]), 0)
    d = np.abs(img.astype(np.float32) - bg).max(-1)
    m = cv2.morphologyEx((d > 18).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return m.astype(bool)


def main():
    mesh_p, views_dir, restyle_dir, source_p, out = sys.argv[1:6]
    size = int(sys.argv[6]) if len(sys.argv) > 6 else 4096
    suffix = sys.argv[7] if len(sys.argv) > 7 else "_s1_d82"
    os.makedirs(out, exist_ok=True)
    d = np.load(mesh_p)
    P = torch.tensor(d["positions"], device=DEV)
    F = torch.tensor(d["faces"], dtype=torch.long, device=DEV)
    N = torch.tensor(d["normals"], device=DEV)
    UV = torch.tensor(d["uvs"], device=DEV)
    meta = json.load(open(os.path.join(views_dir, "views.json")))
    S = meta["size"]

    fid, bary = uv_rasterize(UV, F, size)
    tm = fid >= 0
    tri = F[fid[tm]]
    b = bary[tm].unsqueeze(-1)
    pos = (P[tri] * b).sum(1)
    nrm = (N[tri] * b).sum(1)
    nrm = nrm / nrm.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    T = pos.shape[0]
    print("texels on the surface", T, flush=True)

    # the views: (name, yaw, grey image in the camera's square frame, usable mask, weight)
    src = np.asarray(Image.open(source_p).convert("RGBA"))
    x0, y0, cs = CROP
    sq = np.zeros((cs, cs, 4), np.uint8)
    ys, xs = slice(max(0, -y0), min(cs, src.shape[0] - y0)), slice(max(0, -x0), min(cs, src.shape[1] - x0))
    sq[ys, xs] = src[ys.start + y0:ys.stop + y0, xs.start + x0:xs.stop + x0]
    sq = cv2.resize(sq, (S, S), interpolation=cv2.INTER_AREA)
    src_grey = grey(sq[..., :3])
    src_mask = sq[..., 3] > 200
    views = [("source", 0.0, src_grey, src_mask, SOURCE_WEIGHT)]
    for v in meta["views"]:
        if v["yaw"] == 0:
            continue
        p = os.path.join(restyle_dir, f"{v['name']}{suffix}.png")
        if not os.path.exists(p):
            print("no restyled view", p)
            continue
        img = np.asarray(Image.open(p).convert("RGB").resize((S, S), Image.LANCZOS))
        g = grey(img)
        # started from the render (img2img), the view keeps the render's outline: its own alpha says where the statue is
        m = np.asarray(Image.open(os.path.join(views_dir, f"{v['name']}_alpha.png"))) > 127
        g = match_hist(g[m], src_grey[src_mask])(g)  # the source's tones on the statue
        views.append((v["name"], v["yaw"], g.astype(np.float32), m, 1.0))

    acc = torch.zeros(T, device=DEV)
    wsum = torch.zeros(T, device=DEV)
    wsrc = torch.zeros(T, device=DEV)
    for name, yaw, g, m, weight in views:
        cam = camera(yaw)
        depth = torch.tensor(np.load(os.path.join(views_dir, f"yaw{int(yaw):03d}_depth.npy")), device=DEV)
        r = EDGE_ERODE["source" if name == "source" else "restyled"]
        usable = torch.tensor(cv2.erode(m.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)).astype(bool), device=DEV)
        u, v, dz = project(pos, cam, S)
        ui = u.floor().long().clamp(0, S - 1)
        vi = v.floor().long().clamp(0, S - 1)
        inside = (u >= 0) & (u < S) & (v >= 0) & (v < S)
        dmap = depth[vi, ui]
        visible = inside & torch.isfinite(dmap) & (dz <= dmap * 1.004 + 2e-3) & usable[vi, ui]
        view_dir = cam["eye"].to(DEV) - pos
        view_dir = view_dir / view_dir.norm(dim=-1, keepdim=True)
        facing = (nrm * view_dir).sum(-1).clamp(min=0)
        w = visible.float() * facing ** FACING_POWER * weight
        # bilinear sample of the view
        gt = torch.tensor(g, device=DEV)[None, None]
        grid = torch.stack([u / S * 2 - 1, v / S * 2 - 1], -1)[None, None]
        val = Fn.grid_sample(gt, grid, mode="bilinear", align_corners=False)[0, 0, 0]
        acc += w * val
        wsum += w
        if name == "source":
            wsrc = w
        print(f"{name:8s} yaw {yaw:5.1f}: covers {(w > 0).float().mean().item():.3f} of the surface", flush=True)

    # what no view saw: Pixal3D's own base colour (tools/fallback.py), given the source's tones; a weight
    # so small it only decides where the views are silent
    np.save(os.path.join(out, "seen.npy"), (wsum > 0).cpu().numpy())
    fb_dir = os.environ.get("BAKE_FALLBACK")
    if fb_dir:
        fb = np.load(os.path.join(fb_dir, "fallback.npy"))
        needed = torch.tensor(np.load(os.path.join(fb_dir, "needed.npy")), device=DEV)
        ok = np.isfinite(fb)
        fbm = np.zeros_like(fb)
        fbm[ok] = match_hist(fb[ok], src_grey[src_mask])(fb[ok])
        fbt = torch.tensor(fbm, device=DEV)
        okt = torch.tensor(ok, device=DEV)
        print(f"visible from the allowed poses: {needed.float().mean().item():.4f} of the texels; of those, seen by a view: "
              f"{((wsum > 0) & needed).float().sum().item() / needed.float().sum().item():.4f}, by none: "
              f"{((wsum == 0) & needed).float().sum().item() / needed.float().sum().item():.4f}", flush=True)
        w_fb = okt.float() * 0.01
        acc += w_fb * fbt
        wsum += w_fb
    covered = wsum > 0
    print(f"covered by some view: {covered.float().mean().item():.4f}; by the source: {(wsrc > 0).float().mean().item():.4f}", flush=True)
    tex = np.zeros((size, size), np.float32)
    have = np.zeros((size, size), np.uint8)
    vals = (acc / wsum.clamp(min=1e-8)).cpu().numpy()
    tm_np = tm.cpu().numpy()
    cov = covered.cpu().numpy()
    tex[tm_np] = np.where(cov, vals, 0)
    have[tm_np] = cov.astype(np.uint8)
    # what no view saw, and the gutters between UV charts: filled from the neighbours in UV space
    u8 = (np.clip(tex, 0, 1) * 255).round().astype(np.uint8)
    holes = (have == 0).astype(np.uint8)
    filled = cv2.inpaint(u8, holes, 5, cv2.INPAINT_TELEA)
    Image.fromarray(filled).save(os.path.join(out, "albedo.png"))
    Image.fromarray((have * 255).astype(np.uint8)).save(os.path.join(out, "coverage.png"))
    json.dump({"texels": int(T), "covered": float(covered.float().mean()), "by_source": float((wsrc > 0).float().mean()),
               "views": [v[0] for v in views]}, open(os.path.join(out, "bake.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
