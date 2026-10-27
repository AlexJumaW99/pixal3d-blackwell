"""Silhouettes of Pixal3D generations against the multi-view inputs, through Pixal3D's own projection.

    python /tools/mv_compare.py <input_dir> <name>=<gen.pt> [<name>=<gen.pt> ...]

For each frame of <input_dir>/transforms.json (the cameras tools/mv_prep.py derived) and each generation: the IoU of
the generation's projected silhouette (raw vertices placed with ProjGrid's grid, as tools/mv_prep.py does) against
that frame's alpha, and between the first two generations. Frame 0's alpha is the photo's own matte; the others
are the clay silhouettes the restyled views were drawn on.
"""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/tools")
import mv_prep as mp  # noqa: E402


def world_points(pt):
    d = torch.load(pt, map_location="cpu")
    v = d["vertices"].double().numpy()
    coords = d["coords"].double().numpy()
    if coords.shape[1] == 4:
        coords = coords[:, 1:]
    P = mp.match_axes(v, (coords + 0.5) / d["res"] - 0.5)
    return torch.tensor((mp.GRID_R @ P @ v.T).T, dtype=torch.float32, device="cuda")


def main():
    src, gens = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
    meta = json.load(open(os.path.join(src, "transforms.json")))
    fov, size = meta["camera_angle_x"], 1024
    pts = {name: world_points(pt) for name, pt in gens}
    names = [n for n, _ in gens]
    head = f"{'frame':<8}" + "".join(f"{n + ' vs input':>18}" for n in names)
    if len(names) > 1:
        head += f"{names[0] + ' vs ' + names[1]:>18}"
    print(head)
    for fr in meta["frames"]:
        alpha = Image.open(os.path.join(src, fr["file_path"])).getchannel(3).resize((size, size), Image.Resampling.LANCZOS)
        target = torch.tensor(np.asarray(alpha) > 127, device="cuda")
        c2w = np.array(fr["transform_matrix"])
        sil = {n: mp.silhouette(p, c2w, fov, size) for n, p in pts.items()}
        row = f"{fr['name']:<8}" + "".join(f"{mp.iou(sil[n], target):>18.3f}" for n in names)
        if len(names) > 1:
            row += f"{mp.iou(sil[names[0]], sil[names[1]]):>18.3f}"
        print(row)


if __name__ == "__main__":
    main()
