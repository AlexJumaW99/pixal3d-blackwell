"""Renders of Pixal3D generations through the cameras of a multi-view input, next to the input views.

    python /tools/render_check.py <input_dir> <out_dir> <name>=<gen.pt> [<name>=<gen.pt> ...] [--size 1024]

For each generation and each frame of <input_dir>/transforms.json, through that frame's camera with Pixal3D's
projection (tools/views.py's rasterizer, its FOV rebound to the input's): a clay render, an albedo render (the
generation's base_color sampled at its vertices the way tools/uvbake.py samples it for the texture, lit like the
clay), the silhouette IoU against the frame's alpha, and a sheet: photo (on black, as Pixal3D sees it) | albedo |
clay | the photo's edges over the clay. The mesh is the raw generation placed in the projection frame
(tools/frames.py). Writes <out_dir>/<name>_<frame>_*.png and <out_dir>/check.json.
"""
import json
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/work")
sys.path.insert(0, "/tools")
import views  # noqa: E402
from frames import world_from_raw  # noqa: E402


def load_gen(pt):
    from flex_gemm.ops.grid_sample import grid_sample_3d
    d = torch.load(pt, map_location="cpu")
    v = d["vertices"].float().cuda()
    f = d["faces"].long().cuda()
    coords = d["coords"].cuda()
    if coords.shape[1] == 3:
        coords = torch.cat([torch.zeros_like(coords[:, :1]), coords], -1)
    res = d["res"]
    attrs = d["attrs"].float().cuda()
    lo, hi = d["layout"]["base_color"]
    col = grid_sample_3d(attrs, coords.int(), shape=torch.Size([1, attrs.shape[1], res, res, res]),
                         grid=((v + 0.5) * res).reshape(1, -1, 3), mode="trilinear").reshape(-1, attrs.shape[1])[:, lo:hi].clamp(0, 1)
    vc = v.double().cpu().numpy()
    cc = d["coords"].double().numpy()
    if cc.shape[1] == 4:
        cc = cc[:, 1:]
    world = torch.tensor(world_from_raw(vc, cc, res)[0], dtype=torch.float32, device="cuda")
    fn = torch.linalg.cross(world[f[:, 1]] - world[f[:, 0]], world[f[:, 2]] - world[f[:, 0]])  # area-weighted
    n = torch.zeros_like(world).index_add_(0, f.reshape(-1), fn.repeat_interleave(3, 0))
    n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return world, f, n, col


def render(world, f, n, col, c2w, size):
    c2w = torch.tensor(c2w, dtype=torch.float32, device="cuda")
    cam = {"eye": c2w[:3, 3], "x": c2w[:3, 0], "y": c2w[:3, 1], "z": c2w[:3, 2]}
    fid, bary, _ = views.rasterize(world, f, cam, size)
    m = fid >= 0
    tri = f[fid[m]]
    b = bary[m].unsqueeze(-1)
    nrm = (n[tri] * b).sum(1)
    nrm = nrm / nrm.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    view = cam["eye"] - (world[tri] * b).sum(1)
    view = view / view.norm(dim=-1, keepdim=True)
    nrm = torch.where((nrm * view).sum(-1, keepdim=True) < 0, -nrm, nrm)
    # the lighting of views.clay: key from upper camera-left, fill from camera-right
    key = -0.55 * cam["x"] + 0.65 * cam["y"] + 0.52 * cam["z"]
    key = key / key.norm()
    fill = 0.6 * cam["x"] + 0.1 * cam["y"] + 0.8 * cam["z"]
    fill = fill / fill.norm()
    lam = (0.08 + 0.72 * (nrm @ key).clamp(min=0) + 0.22 * (nrm @ fill).clamp(min=0))
    h = (key + view) / (key + view).norm(dim=-1, keepdim=True)
    spec = 0.35 * (nrm * h).sum(-1).clamp(min=0) ** 24
    clay = torch.zeros(size, size, device="cuda")
    clay[m] = (lam + spec).clamp(0, 1)
    alb = torch.zeros(size, size, 3, device="cuda")
    alb[m] = ((col[tri] * b).sum(1) * (0.35 + 0.65 * lam[:, None]) + 0.5 * spec[:, None]).clamp(0, 1)
    return clay.cpu().numpy(), alb.cpu().numpy(), m.cpu().numpy()


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    size = int(sys.argv[sys.argv.index("--size") + 1]) if "--size" in sys.argv else 1024
    if "--size" in sys.argv:
        args.remove(str(size))
    inp, out, gens = args[0], args[1], [a.split("=", 1) for a in args[2:]]
    os.makedirs(out, exist_ok=True)
    meta = json.load(open(os.path.join(inp, "transforms.json")))
    fov = float(meta["camera_angle_x"])
    views.project.__defaults__ = (fov,)  # views.py binds the Amina run's FOV as project()'s default
    result = {}
    for name, pt in gens:
        world, f, n, col = load_gen(pt)
        print(f"[{name}] {len(world)} vertices, {len(f)} faces; base colour mean {col.mean(0).cpu().numpy().round(3)}", flush=True)
        for fr in meta["frames"]:
            fov_f = float(fr.get("camera_angle_x", fov))
            views.project.__defaults__ = (fov_f,)
            clay, alb, m = render(world, f, n, col, fr["transform_matrix"], size)
            img = Image.open(os.path.join(inp, fr["file_path"])).convert("RGBA").resize((size, size), Image.Resampling.LANCZOS)
            x = np.asarray(img).astype(np.float32) / 255
            photo = x[..., :3] * x[..., 3:]
            a = x[..., 3] > 0.5
            iou = float((a & m).sum() / (a | m).sum())
            edges = cv2.Canny((photo.mean(-1) * 255).astype(np.uint8), 30, 80) > 0
            ov = np.repeat(clay[..., None], 3, -1)
            ov[edges] = (1.0, 0.15, 0.1)
            tag = f"{name}_{fr.get('name', os.path.splitext(fr['file_path'])[0])}"
            to8 = lambda t: (np.clip(t, 0, 1) * 255).round().astype(np.uint8)
            Image.fromarray(to8(alb)).save(os.path.join(out, f"{tag}_albedo.png"))
            Image.fromarray(to8(clay)).save(os.path.join(out, f"{tag}_clay.png"))
            sheet = np.concatenate([photo, alb, np.repeat(clay[..., None], 3, -1), ov], 1)
            Image.fromarray(to8(cv2.resize(sheet, (sheet.shape[1] // 2, sheet.shape[0] // 2), interpolation=cv2.INTER_AREA))).save(
                os.path.join(out, f"{tag}_sheet.jpg"), quality=92)
            result[tag] = {"silhouette_iou": iou}
            print(f"[{tag}] silhouette IoU vs the view's alpha {iou:.4f}", flush=True)
        del world, f, n, col
        torch.cuda.empty_cache()
    json.dump(result, open(os.path.join(out, "check.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
