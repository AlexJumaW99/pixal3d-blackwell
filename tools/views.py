"""Cameras, a z-buffered rasterizer and clay renders for the Amina statue, in the GLB's frame (Y up).

The canonical camera is Pixal3D's (out/sv1024/camera.json): at (0, 0, -D) looking at the origin, up +Y,
horizontal FOV 30.77° over a square frame that is Pixal3D's crop of the source. Other views turn the
camera about the vertical axis (yaw) and tilt it (pitch), as three.js's lookAt does (harness: index.html).

    python /tools/views.py <mesh.npz> <out_dir> [size=1024] [yaws=45,90,...]
"""
import json
import math
import sys

import numpy as np
import torch
from PIL import Image

FOV = 0.5370941572530058
DIST = 1.8168962001800537


def camera(yaw_deg=0.0, pitch_deg=0.0, dist=DIST):
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    eye = torch.tensor([-dist * math.sin(y) * math.cos(p), dist * math.sin(p), -dist * math.cos(y) * math.cos(p)])
    z = eye / eye.norm()  # three.js: the camera looks down its -z, z points from the target to the eye
    up = torch.tensor([0.0, 1.0, 0.0])
    x = torch.linalg.cross(up, z)
    x = x / x.norm()
    yv = torch.linalg.cross(z, x)
    return {"eye": eye, "x": x, "y": yv, "z": z}


def project(P, cam, size, fov=FOV):
    """world (N,3) -> pixel (u right, v down), view depth (positive in front)"""
    d = P - cam["eye"].to(P)
    xc, yc, zc = d @ cam["x"].to(P), d @ cam["y"].to(P), d @ cam["z"].to(P)
    f = (size / 2) / math.tan(fov / 2)
    depth = -zc
    u = size / 2 + f * xc / depth
    v = size / 2 - f * yc / depth
    return u, v, depth


@torch.no_grad()
def rasterize(P, F, cam, size, chunk=150_000):
    """z-buffer: per pixel the nearest face id (-1 if none), its screen barycentrics and view depth"""
    u, v, dep = project(P, cam, size)
    dev = P.device
    tri_u, tri_v, tri_d = u[F], v[F], dep[F]
    best = torch.full((size * size,), float("inf"), device=dev)
    cands = []
    for f0 in range(0, F.shape[0], chunk):
        tu, tv, td = tri_u[f0:f0 + chunk], tri_v[f0:f0 + chunk], tri_d[f0:f0 + chunk]
        lo_u = torch.floor(tu.min(1).values - 0.5).clamp(0, size - 1).long()
        hi_u = torch.ceil(tu.max(1).values - 0.5).clamp(0, size - 1).long()
        lo_v = torch.floor(tv.min(1).values - 0.5).clamp(0, size - 1).long()
        hi_v = torch.ceil(tv.max(1).values - 0.5).clamp(0, size - 1).long()
        w = (hi_u - lo_u + 1).clamp(min=0)
        h = (hi_v - lo_v + 1).clamp(min=0)
        n = w * h
        fidx = torch.repeat_interleave(torch.arange(tu.shape[0], device=dev), n)
        start = torch.cumsum(n, 0) - n
        local = torch.arange(int(n.sum()), device=dev) - start[fidx]
        px = lo_u[fidx] + local % w[fidx]
        py = lo_v[fidx] + local // w[fidx]
        cx, cy = px.float() + 0.5, py.float() + 0.5
        ax, ay = tu[fidx, 0], tv[fidx, 0]
        bx, by = tu[fidx, 1], tv[fidx, 1]
        qx, qy = tu[fidx, 2], tv[fidx, 2]
        den = (bx - ax) * (qy - ay) - (qx - ax) * (by - ay)
        ok = den.abs() > 1e-12
        den = torch.where(ok, den, torch.ones_like(den))
        l1 = ((cx - ax) * (qy - ay) - (qx - ax) * (cy - ay)) / den
        l2 = ((bx - ax) * (cy - ay) - (cx - ax) * (by - ay)) / den
        l0 = 1 - l1 - l2
        inside = ok & (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6) & (td[fidx].min(1).values > 0)
        # perspective-correct depth: 1/z is linear on screen
        inv = l0 / td[fidx, 0] + l1 / td[fidx, 1] + l2 / td[fidx, 2]
        z = 1 / inv
        pix = (py * size + px)[inside]
        zi = z[inside]
        best.scatter_reduce_(0, pix, zi, reduce="amin")
        cands.append((pix, zi, (fidx + f0)[inside], torch.stack([l0, l1, l2], -1)[inside]))
    fid = torch.full((size * size,), -1, dtype=torch.long, device=dev)
    bary = torch.zeros((size * size, 3), device=dev)
    for pix, zi, fi, b in cands:
        win = zi <= best[pix]
        fid[pix[win]] = fi[win]
        bary[pix[win]] = b[win]
    return fid.view(size, size), bary.view(size, size, 3), best.view(size, size)


def clay(P, F, N, cam, size):
    """a grey clay render (key light from upper camera-left, as the source is lit) and its alpha"""
    fid, bary, depth = rasterize(P, F, cam, size)
    m = fid >= 0
    nrm = (N[F[fid[m]]] * bary[m].unsqueeze(-1)).sum(1)
    nrm = nrm / nrm.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    view = cam["eye"].to(P) - (P[F[fid[m]]] * bary[m].unsqueeze(-1)).sum(1)
    view = view / view.norm(dim=-1, keepdim=True)
    nrm = torch.where((nrm * view).sum(-1, keepdim=True) < 0, -nrm, nrm)
    key = (-0.55 * cam["x"] + 0.65 * cam["y"] + 0.52 * cam["z"]).to(P)
    key = key / key.norm()
    fill = (0.6 * cam["x"] + 0.1 * cam["y"] + 0.8 * cam["z"]).to(P)
    fill = fill / fill.norm()
    lam = 0.08 + 0.72 * (nrm @ key).clamp(min=0) + 0.22 * (nrm @ fill).clamp(min=0)
    h = (key + view) / (key + view).norm(dim=-1, keepdim=True)
    spec = 0.35 * (nrm * h).sum(-1).clamp(min=0) ** 24
    img = torch.zeros(size, size, device=P.device)
    img[m] = (lam + spec).clamp(0, 1)
    return img, m, depth


def main():
    src, out = sys.argv[1], sys.argv[2]
    size = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
    yaws = [float(y) for y in (sys.argv[4] if len(sys.argv) > 4 else "0,45,90,135,180,225,270,315").split(",")]
    d = np.load(src)
    P = torch.tensor(d["positions"], device="cuda")
    F = torch.tensor(d["faces"], dtype=torch.long, device="cuda")
    N = torch.tensor(d["normals"], device="cuda")
    meta = {"fov": FOV, "dist": DIST, "size": size, "views": []}
    import os
    os.makedirs(out, exist_ok=True)
    for yaw in yaws:
        cam = camera(yaw)
        img, m, depth = clay(P, F, N, cam, size)
        grey = (img.cpu().numpy() * 255).astype(np.uint8)
        a = (m.cpu().numpy() * 255).astype(np.uint8)
        name = f"yaw{int(yaw):03d}"
        rgb = np.where(a[..., None] > 0, np.repeat(grey[..., None], 3, -1), 128).astype(np.uint8)
        Image.fromarray(rgb).save(f"{out}/{name}_clay.png")
        Image.fromarray(a).save(f"{out}/{name}_alpha.png")
        np.save(f"{out}/{name}_depth.npy", depth.cpu().numpy().astype(np.float32))
        meta["views"].append({"name": name, "yaw": yaw, "pitch": 0.0})
        print(name, "coverage", float(m.float().mean()))
    json.dump(meta, open(f"{out}/views.json", "w"), indent=2)


if __name__ == "__main__":
    main()
