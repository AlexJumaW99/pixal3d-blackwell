"""Pixal3D multi-view input built from the Amina single-view run: frame 0, cameras (checked), views.

    python /tools/mv_prep.py crop    <rgba.png> <sv_dir> <out.png>
    python /tools/mv_prep.py cameras <sv.pt> <mesh.npz> <views_dir> <frame0.png> <cameras.json> [yaws=0,90,180,270]
    python /tools/mv_prep.py build   <frame0.png> <restyle_dir> <views_dir> <cameras.json> <out_dir> [yaws=90,180,270]

crop: frame 0 is the photo exactly as the single-view run saw it: preprocess_image's square crop (1.1 x the
  alpha's bounding box), kept as RGBA because inference_mv takes the mask from the alpha. Checked against the
  run's own preprocessed.png.
cameras: a camera-to-world matrix for each views.py yaw, in the frame Pixal3D projects image features in (Z up,
  front camera at (0, -d, 0)). Nothing about that frame is assumed: the generation's voxel coords are placed
  where ProjGrid puts them, the raw vertices are matched to the coords axis by axis, and the mesh views.py
  rendered (webmesh's npz) is matched to the raw vertices the same way. Not run_single's ROT: the GLB as loaded
  back is not ROT @ raw (it is the raw mesh turned 180 degrees about its up axis). Each camera is then checked
  by projecting the mesh with Pixal3D's own projection against views.py's silhouette at that yaw (and the
  photo's at yaw 0), next to the mirrored yaw.
build: the restyled renders with their clay silhouettes as alpha, frame 0 and transforms.json.
"""
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, "/work")

GRID_R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)  # ProjGrid: grid index frame -> projection frame


def front(d):
    """ProjGrid's front_view_transform_matrix with the camera at distance d"""
    return np.array([[1, 0, 0, 0], [0, 0, -1, -d], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)


def crop_like_pixal3d(img):
    """Pixal3DImageTo3DPipeline.preprocess_image up to the crop, without compositing onto black"""
    scale = min(1, 1024 / max(img.size))
    if scale < 1:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
    alpha = np.array(img)[:, :, 3]
    bbox = np.argwhere(alpha > 0.8 * 255)
    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = int(max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 1.1)
    return img.crop((center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2))


def cmd_crop(rgba, sv_dir, out):
    img = Image.open(rgba)
    assert img.mode == "RGBA", "the input must be RGBA (its own matte)"
    crop = crop_like_pixal3d(img)
    crop.save(out)
    c = np.asarray(crop).astype(np.float32) / 255
    mine = (np.clip(c[..., :3] * c[..., 3:], 0, 1) * 255).astype(np.uint8).astype(np.int16)
    ref = np.asarray(Image.open(os.path.join(sv_dir, "preprocessed.png")).convert("RGB")).astype(np.int16)
    assert mine.shape == ref.shape, (mine.shape, ref.shape)
    diff = np.abs(mine - ref)
    print(f"[crop] {crop.size} RGBA; composited on black vs the run's preprocessed.png: max diff {diff.max()}, mean {diff.mean():.4f}")


def match_axes(v, p):
    """P with p ~ P @ v: each raw-vertex axis matched to a voxel-index axis by its extent (min and max)"""
    P = np.zeros((3, 3))
    used = set()
    for a in range(3):
        best = min(((abs(s * v[:, a].min() - p[:, b].min()) + abs(s * v[:, a].max() - p[:, b].max()) if s > 0 else
                     abs(-v[:, a].max() - p[:, b].min()) + abs(-v[:, a].min() - p[:, b].max()), b, s)
                    for b in range(3) for s in (1, -1)))
        err, b, s = best
        print(f"[axes] raw {a} -> index {b} sign {s:+d}: extent error {err:.5f}")
        assert b not in used, "two raw axes matched the same index axis"
        used.add(b)
        P[b, a] = s
    return P


def glb_camera(yaw_deg, dist):
    """views.camera(yaw) as a 4x4 camera-to-world in the GLB frame (three.js lookAt: looks down -z, up +y)"""
    y = math.radians(yaw_deg)
    eye = np.array([-dist * math.sin(y), 0.0, -dist * math.cos(y)])
    z = eye / np.linalg.norm(eye)
    x = np.cross([0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x)
    yv = np.cross(z, x)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, yv, z, eye
    return m


def silhouette(points, c2w, fov, size):
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import project_points_to_image_batch
    pts, depth, valid = project_points_to_image_batch(points, torch.tensor(c2w, dtype=torch.float32, device=points.device)[None],
                                                      torch.tensor([fov], dtype=torch.float32, device=points.device), size)
    pts = pts[0][valid[0]].round().long()
    m = torch.zeros(size, size, device=points.device)
    m[pts[:, 1].clamp(0, size - 1), pts[:, 0].clamp(0, size - 1)] = 1
    return F.max_pool2d(m[None, None], 5, 1, 2)[0, 0] > 0  # close the gaps between projected vertices


def iou(a, b):
    return float((a & b).sum() / (a | b).sum())


def cmd_cameras(pt, npz, views_dir, frame0, out, yaws):
    meta = json.load(open(os.path.join(views_dir, "views.json")))
    fov, dist, size = meta["fov"], meta["dist"], meta["size"]
    d = torch.load(pt, map_location="cpu")
    v = d["vertices"].double().numpy()
    coords, res = d["coords"].double().numpy(), d["res"]
    if coords.shape[1] == 4:  # a leading batch column
        coords = coords[:, 1:]
    print(f"[mesh] {len(v)} vertices, {len(coords)} voxels, res {res}, coords range {coords.min(0)} .. {coords.max(0)}")
    p = (coords + 0.5) / res - 0.5  # voxel centres in [-0.5, 0.5], grid index order
    P = match_axes(v, p)
    g = np.load(npz)["positions"].astype(np.float64)
    Q = match_axes(g, v)  # views.py's mesh -> raw vertices
    T = GRID_R @ P @ Q  # views.py frame -> projection frame (ProjGrid: world = R @ index, index in [-0.5, 0.5])
    print(f"[frame] views.py frame -> projection frame:\n{T}\n det {np.linalg.det(T):+.3f}")
    assert abs(np.linalg.det(T) - 1) < 1e-9, "views.py's frame and the projection frame differ by a mirror"
    T4 = np.eye(4)
    T4[:3, :3] = T
    world = torch.tensor((GRID_R @ P @ v.T).T, dtype=torch.float32, device="cuda")
    photo = np.asarray(Image.open(frame0).getchannel(3).resize((size, size), Image.Resampling.LANCZOS)) > 127
    cams = {}
    for yaw in yaws:
        c2w = T4 @ glb_camera(yaw, dist)
        if yaw == 0:
            err = np.abs(c2w - front(dist)).max()
            print(f"[yaw   0] views.py's front camera vs Pixal3D's front_view_transform_matrix: max diff {err:.2e}")
            assert err < 1e-6
        clay = torch.tensor(np.asarray(Image.open(os.path.join(views_dir, f"yaw{int(yaw):03d}_alpha.png"))) > 127, device="cuda")
        own = iou(silhouette(world, c2w, fov, size), clay)
        mirrored = iou(silhouette(world, T4 @ glb_camera(-yaw, dist), fov, size), clay)
        line = f"[yaw {int(yaw):3d}] silhouette IoU vs views.py: {own:.3f} (mirrored yaw {-int(yaw) % 360}: {mirrored:.3f})"
        if yaw == 0:
            line += f"; vs the photo's matte: {iou(silhouette(world, c2w, fov, size), torch.tensor(photo, device='cuda')):.3f}"
        print(line)
        cams[str(int(yaw))] = c2w.tolist()
    json.dump({"fov": fov, "dist": dist, "size": size, "views_to_projection": T.tolist(), "c2w": cams}, open(out, "w"), indent=2)


def cmd_build(frame0, restyle_dir, views_dir, cameras, out, yaws):
    cams = json.load(open(cameras))
    os.makedirs(out, exist_ok=True)
    frames = [{"file_path": "view00_yaw000.png", "name": "yaw000", "transform_matrix": front(cams["dist"]).tolist()}]
    Image.open(frame0).save(os.path.join(out, frames[0]["file_path"]))
    for i, yaw in enumerate(yaws, 1):
        name = f"yaw{int(yaw):03d}"
        rgb = Image.open(os.path.join(restyle_dir, f"{name}_s1_d82.png")).convert("RGB")
        alpha = Image.open(os.path.join(views_dir, f"{name}_alpha.png")).convert("L")
        if rgb.size != alpha.size:
            print(f"[{name}] restyle {rgb.size} -> {alpha.size}")
            rgb = rgb.resize(alpha.size, Image.Resampling.LANCZOS)
        # how well the restyle kept the clay outline: its own foreground (away from the plain background) vs the clay
        a = np.asarray(alpha) > 127
        x = np.asarray(rgb).astype(np.float32)
        bg = np.median(x[~a], axis=0)
        fg = np.abs(x - bg).max(-1) > 24
        print(f"[{name}] background {bg.round()}; restyle foreground vs clay silhouette IoU {float((fg & a).sum() / (fg | a).sum()):.3f}, "
              f"clay pixels that look like background {float((~fg & a).sum() / a.sum()):.3f}")
        rgba = rgb.copy()
        rgba.putalpha(alpha)
        f = f"view{i:02d}_{name}.png"
        rgba.save(os.path.join(out, f))
        frames.append({"file_path": f, "name": name, "transform_matrix": cams["c2w"][str(int(yaw))]})
    json.dump({"camera_angle_x": cams["fov"], "mesh_scale": 1.0, "frames": frames}, open(os.path.join(out, "transforms.json"), "w"), indent=2)
    print(f"[build] {len(frames)} frames -> {out}/transforms.json")


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "crop":
        cmd_crop(*args)
    elif cmd == "cameras":
        cmd_cameras(*args[:5], [float(y) for y in (args[5] if len(args) > 5 else "0,90,180,270").split(",")])
    elif cmd == "build":
        cmd_build(*args[:5], [float(y) for y in (args[5] if len(args) > 5 else "90,180,270").split(",")])
    else:
        sys.exit(__doc__)
