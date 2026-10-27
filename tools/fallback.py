"""What the bake needs beyond its views, for the Amina statue.

    python /tools/fallback.py <mesh.npz> <raw.pt> <out_dir> [size=4096]

1. needed.npy: the texels a visitor can see, from any of the poses the page allows (yaw all round, pitch
   -10..+16 degrees): depth-tested from 24 x 4 cameras.
2. fallback.npy: Pixal3D's own base colour for every texel (as grey), sampled from its raw voxel volume
   (trilinear, as o_voxel's to_glb samples it), for what no view saw.
The volume is in Pixal3D's frame: a GLB point (x, y, z) is (-x, y, -z) there (to_glb's axis swap then
inference.py's rotation); checked by how many texels land in occupied voxels.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/tools")
from uvbake import uv_rasterize  # noqa: E402
from views import camera, project, rasterize  # noqa: E402

DEV = os.environ.get("BAKE_DEVICE", "cuda")


def main():
    mesh_p, raw_p, out = sys.argv[1:4]
    size = int(sys.argv[4]) if len(sys.argv) > 4 else 4096
    os.makedirs(out, exist_ok=True)
    d = np.load(mesh_p)
    P = torch.tensor(d["positions"], device=DEV)
    F = torch.tensor(d["faces"], dtype=torch.long, device=DEV)
    N = torch.tensor(d["normals"], device=DEV)
    UV = torch.tensor(d["uvs"], device=DEV)
    fid, bary = uv_rasterize(UV, F, size)
    tm = fid >= 0
    tri = F[fid[tm]]
    b = bary[tm].unsqueeze(-1)
    pos = (P[tri] * b).sum(1)
    nrm = (N[tri] * b).sum(1)
    nrm = nrm / nrm.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    # 1. seen from the poses the page allows
    S = 512
    needed = torch.zeros(pos.shape[0], dtype=torch.bool, device=DEV)
    for pitch in (-10, 0, 8, 16):
        for yaw in range(0, 360, 15):
            cam = camera(yaw, pitch)
            _, _, depth = rasterize(P, F, cam, S)
            u, v, dz = project(pos, cam, S)
            ui, vi = u.floor().long().clamp(0, S - 1), v.floor().long().clamp(0, S - 1)
            inside = (u >= 0) & (u < S) & (v >= 0) & (v < S)
            dmap = depth[vi, ui]
            vd = cam["eye"].to(DEV) - pos
            facing = (nrm * vd).sum(-1) / vd.norm(dim=-1)
            needed |= inside & torch.isfinite(dmap) & (dz <= dmap + 2e-3) & (facing > 0.05)
    print(f"texels {pos.shape[0]}, seen from the allowed poses {needed.float().mean().item():.4f}", flush=True)

    # 2. Pixal3D's base colour at every texel
    raw = torch.load(raw_p, map_location="cpu")
    res = int(raw["res"])
    coords = raw["coords"].long().numpy()
    coords = coords[:, -3:]  # (x, y, z): the batch column, if any, dropped
    attrs = raw["attrs"].float().numpy()
    lay = raw["layout"]
    bc = attrs[:, lay["base_color"][0]:lay["base_color"][1]]
    lin = (coords[:, 0] * res + coords[:, 1]) * res + coords[:, 2]
    order = np.argsort(lin)
    lin_s = lin[order]
    q = pos.cpu().numpy() * np.array([-1.0, 1.0, -1.0])
    g = (q + 0.5) * res - 0.5  # voxel centres at integer + 0.5
    g0 = np.floor(g).astype(np.int64)
    fr = g - g0
    acc = np.zeros((q.shape[0], 3))
    wsum = np.zeros(q.shape[0])
    for dx in (0, 1):
        for dy in (0, 1):
            for dz_ in (0, 1):
                c = g0 + np.array([dx, dy, dz_])
                ok = np.all((c >= 0) & (c < res), 1)
                key = (c[:, 0] * res + c[:, 1]) * res + c[:, 2]
                idx = np.searchsorted(lin_s, key)
                idx = np.clip(idx, 0, len(lin_s) - 1)
                hit = ok & (lin_s[idx] == key)
                w = (fr[:, 0] if dx else 1 - fr[:, 0]) * (fr[:, 1] if dy else 1 - fr[:, 1]) * (fr[:, 2] if dz_ else 1 - fr[:, 2])
                w = np.where(hit, w, 0)
                acc += w[:, None] * bc[order[idx]]
                wsum += w
    print(f"texels in occupied voxels: {(wsum > 0).mean():.4f} (axis mapping check)", flush=True)
    col = acc / np.maximum(wsum, 1e-8)[:, None]
    grey = 0.2126 * col[:, 0] + 0.7152 * col[:, 1] + 0.0722 * col[:, 2]
    np.save(os.path.join(out, "needed.npy"), needed.cpu().numpy())
    np.save(os.path.join(out, "fallback.npy"), np.where(wsum > 0, grey, np.nan).astype(np.float32))


if __name__ == "__main__":
    main()
