"""Pixal3D two-view input for the marble bust: the front photo is frame 0, the 3/4 photo's camera is fitted.

    python /tools/fit_view.py crop  <front_bust_rgba.png> <tq_bust_rgba.png> <out_dir>
    python /tools/fit_view.py fit   <sv_dir> <seed> <out_dir>
    python /tools/fit_view.py landmarks <sv_dir> <seed> <out_dir> [mesh.pt]
    python /tools/fit_view.py build <sv_dir> <out_dir>
    python /tools/fit_view.py score <input_dir> <name>=<gen.pt> [...]

crop: both views cut the way Pixal3D's preprocess_image cuts (a square 1.1 x the bounding box of alpha > 0.8),
  but from the full-resolution image instead of its 1024 px copy (inference_mv resizes each view itself), kept
  RGBA because inference_mv takes the mask from the alpha. Writes view00_front.png, view01_tq.png, crops.json.
fit: the single-view run's mesh in the frame Pixal3D projects features in (tools/mv_prep.py: voxel coords placed
  where ProjGrid puts them, raw vertices matched to them axis by axis), checked through the canonical front
  camera against frame 0's alpha. Then the 3/4 camera: yaw and pitch about the object's centre, roll, distance
  and an offset in the image plane (the crop need not be centred on the object's origin), scored by silhouette
  IoU against the 3/4 alpha above its neck, since the two photos' bust bottoms are different carvings. A grid
  over yaw x pitch with scale and offset found in image space, then Nelder-Mead on a smoothed IoU from the best
  few, then the mirrored yaw for comparison. Writes fit.json and overlays.
build: transforms.json from fit.json and the single-view run's camera.json.
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
sys.path.insert(0, "/tools")

S = 512             # silhouette resolution
UPPER = 0.68        # scored rows: down to this fraction of the 3/4 alpha's bounding box (above the bust's base)


def crop_full(img):
    a = np.asarray(img)[..., 3]
    ys, xs = np.nonzero(a > 0.8 * 255)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    size = int(max(x1 - x0, y1 - y0) * 1.1)
    box = (cx - size // 2, cy - size // 2, cx + size // 2, cy + size // 2)
    return img.crop(box), [float(b) for b in box]


def cmd_crop(front, tq, out):
    os.makedirs(out, exist_ok=True)
    info = {}
    for name, path in (("view00_front.png", front), ("view01_tq.png", tq)):
        img = Image.open(path)
        assert img.mode == "RGBA"
        c, box = crop_full(img)
        c.save(os.path.join(out, name))
        info[name] = {"source": path, "source_size": img.size, "box": box, "size": c.size,
                      "box_normalised": [box[0] / img.width, box[1] / img.height, box[2] / img.width, box[3] / img.height]}
        print(f"[crop] {name}: {img.size} -> box {np.round(box, 1).tolist()} -> {c.size}")
    json.dump(info, open(os.path.join(out, "crops.json"), "w"), indent=2)


def rot(axis, deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    raise ValueError(axis)


def camera(yaw, pitch, roll, d, u, v):
    """c2w: the canonical front camera (0, -d, 0) moved by (u, v) in its own image plane and rolled about its axis,
    then orbited about the object's centre: pitch about world x, yaw about world z (up)"""
    from mv_prep import front
    orbit = np.eye(4)
    orbit[:3, :3] = rot("z", yaw) @ rot("x", pitch)
    local = np.eye(4)
    local[:3, :3] = rot("z", roll)  # camera z is its viewing axis
    local[:3, 3] = (u, v, 0)
    return orbit @ front(d) @ local


def render(world, c2w, fov, size=S):
    """mesh silhouette through Pixal3D's own projection: vertices splatted, then closed (dilate, erode) so the
    outline is not grown"""
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import project_points_to_image_batch
    pts, _, valid = project_points_to_image_batch(world, torch.tensor(c2w, dtype=torch.float32, device=world.device)[None],
                                                  torch.tensor([fov], dtype=torch.float32, device=world.device), size)
    p = pts[0][valid[0]].floor().long()
    m = torch.zeros(size, size, device=world.device)
    m[p[:, 1], p[:, 0]] = 1
    m = F.max_pool2d(m[None, None], 5, 1, 2)
    m = -F.max_pool2d(-m, 5, 1, 2)
    return m[0, 0]


def iou(a, b, rows=None):
    a, b = a > 0.5, b > 0.5
    if rows is not None:
        a, b = a[:rows], b[:rows]
    return float((a & b).sum() / (a | b).sum())


def soft(m, sigma=2.0):
    k = int(3 * sigma) * 2 + 1
    g = torch.exp(-torch.arange(-(k // 2), k // 2 + 1, dtype=torch.float32, device=m.device) ** 2 / (2 * sigma ** 2))
    g /= g.sum()
    x = F.conv2d(m[None, None], g.view(1, 1, 1, -1), padding=(0, k // 2))
    return F.conv2d(x, g.view(1, 1, -1, 1), padding=(k // 2, 0))[0, 0]


def soft_iou(a, b, rows):
    a, b = a[:rows], b[:rows]
    return float(torch.minimum(a, b).sum() / torch.maximum(a, b).sum())


def warp(m, s, dx, dy):
    """m scaled by s about the image centre, then shifted (dx, dy) px"""
    theta = torch.tensor([[1 / s, 0, -2 * dx / (S * s)], [0, 1 / s, -2 * dy / (S * s)]], dtype=torch.float32, device=m.device)
    grid = F.affine_grid(theta[None], (1, 1, S, S), align_corners=False)
    return F.grid_sample(m[None, None], grid, align_corners=False)[0, 0]


def top_and_cx(m, frac_rows):
    ys = torch.nonzero(m.amax(1) > 0.5)[:, 0]
    top = int(ys.min())
    band = m[top:top + frac_rows]
    xs = torch.nonzero(band > 0.5)[:, 1].float()
    return top, float(xs.mean())


def cmd_fit(sv_dir, seed, out):
    from scipy.optimize import minimize
    from frames import world_from_raw

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cam = json.load(open(os.path.join(sv_dir, "camera.json")))
    fov, d0 = cam["camera_angle_x"], cam["distance"]
    d = torch.load(os.path.join(sv_dir, f"s{seed}.pt"), map_location="cpu")
    v = d["vertices"].double().numpy()
    coords, res = d["coords"].double().numpy(), d["res"]
    if coords.shape[1] == 4:
        coords = coords[:, 1:]
    world_all = torch.tensor(world_from_raw(v, coords, res)[0], dtype=torch.float32, device=dev)
    g = torch.Generator(device="cpu").manual_seed(0)
    world = world_all[torch.randperm(len(world_all), generator=g)[:400_000].to(dev)]
    print(f"[mesh] {len(v)} vertices (400k used in the search), fov {math.degrees(fov):.2f} deg, distance {d0:.4f}, device {dev}", flush=True)

    load_alpha = lambda p: torch.tensor(np.asarray(Image.open(p).getchannel(3).resize((S, S), Image.Resampling.LANCZOS)) / 255.0,
                                        dtype=torch.float32, device=dev)
    a0 = load_alpha(os.path.join(out, "view00_front.png"))
    a1 = load_alpha(os.path.join(out, "view01_tq.png"))

    # frame 0 through the canonical front camera: the frame and the crop must agree before anything is fitted
    r0 = render(world_all, camera(0, 0, 0, d0, 0, 0), fov)
    iou0 = iou(r0, a0)
    print(f"[front] silhouette IoU, canonical front camera vs frame 0's alpha: {iou0:.3f}", flush=True)
    assert iou0 > 0.85, "the mesh does not line up with frame 0 through the front camera: frame or crop convention is off"

    ys = torch.nonzero(a1.amax(1) > 0.5)[:, 0]
    rows = int(ys.min() + UPPER * (ys.max() - ys.min()))
    band = int(0.3 * (ys.max() - ys.min()))
    t_top, t_cx = top_and_cx(a1, band)
    a1s = soft(a1)
    fpx = (S / 2) / math.tan(fov / 2)

    def to_cam(yaw, pitch, s, dx, dy):
        dn = d0 / s
        return yaw, pitch, 0.0, dn, -dx * dn / fpx, dy * dn / fpx

    # grid: for each yaw, pitch the scale s and offset that best match in image space
    cands = []
    for yaw in range(-90, 91, 5):
        for pitch in range(-30, 31, 5):
            r = render(world, camera(yaw, pitch, 0, d0, 0, 0), fov)
            if r.sum() == 0:
                continue
            best = (-1, None)
            for s in np.arange(0.70, 1.41, 0.05):
                rs = warp(r, s, 0, 0)
                if rs.sum() < 10:
                    continue
                top, cx = top_and_cx(rs, band)
                dx, dy = t_cx - cx, t_top - top
                sc = iou(warp(r, s, dx, dy), a1, rows)
                if sc > best[0]:
                    best = (sc, (s, dx, dy))
            cands.append((best[0], yaw, pitch, *best[1]))
    cands.sort(reverse=True)
    print("[grid] best: " + "; ".join(f"yaw {c[1]:+d} pitch {c[2]:+d} s {c[3]:.2f}: IoU {c[0]:.3f}" for c in cands[:5]), flush=True)

    def loss(x):
        r = render(world, camera(*x), fov)
        return 1 - soft_iou(soft(r), a1s, rows)

    results = []
    seen = set()
    for c in cands:
        key = (c[1] // 15, c[2] // 15)
        if key in seen:
            continue
        seen.add(key)
        x0 = np.array(to_cam(c[1], c[2], c[3], c[4], c[5]))
        simplex = np.array([x0] + [x0 + dlt for dlt in np.diag([6.0, 4.0, 3.0, 0.08 * x0[3], 0.03, 0.03])])
        r = minimize(loss, x0, method="Nelder-Mead", options={"initial_simplex": simplex, "xatol": 0.02, "fatol": 1e-5, "maxiter": 600})
        results.append((r.fun, r.x, c))
        print(f"[refine] from yaw {c[1]:+d} pitch {c[2]:+d}: yaw {r.x[0]:+.2f} pitch {r.x[1]:+.2f} roll {r.x[2]:+.2f} "
              f"d {r.x[3]:.4f} u {r.x[4]:+.4f} v {r.x[5]:+.4f}: soft IoU {1 - r.fun:.4f} ({r.nit} it)", flush=True)
        if len(results) == 4:
            break
    results.sort(key=lambda t: t[0])
    x = results[0][1]
    c2w = camera(*x)
    r_best = render(world_all, c2w, fov)
    m = x.copy()
    m[0], m[2], m[4] = -x[0], -x[2], -x[4]
    r_mir = render(world_all, camera(*m), fov)
    fit = {"yaw": x[0], "pitch": x[1], "roll": x[2], "distance": x[3], "u": x[4], "v": x[5], "fov": fov, "d0": d0,
           "c2w": c2w.tolist(), "iou_upper": iou(r_best, a1, rows), "iou_full": iou(r_best, a1),
           "iou_upper_mirrored": iou(r_mir, a1, rows), "iou_front_check": iou0, "scored_rows": rows, "size": S,
           "runs": [{"x": list(map(float, t[1])), "soft_iou": 1 - float(t[0])} for t in results]}
    json.dump(fit, open(os.path.join(out, "fit.json"), "w"), indent=2, default=float)
    print(f"[fit] yaw {x[0]:+.2f} pitch {x[1]:+.2f} roll {x[2]:+.2f} distance {x[3]:.4f} (front {d0:.4f}) offset ({x[4]:+.4f}, {x[5]:+.4f}); "
          f"IoU above the neck {fit['iou_upper']:.3f}, whole {fit['iou_full']:.3f}; mirrored yaw {fit['iou_upper_mirrored']:.3f}", flush=True)

    def overlay(r, a, path):
        r, a = (r > 0.5).cpu().numpy(), (a > 0.5).cpu().numpy()
        img = np.zeros((S, S, 3), np.uint8)
        img[r & a] = (200, 200, 200)
        img[r & ~a] = (230, 60, 60)    # mesh only
        img[~r & a] = (60, 120, 230)   # photo only
        img[rows, :] = (255, 220, 0)
        Image.fromarray(img).save(path)
    overlay(r0, a0, os.path.join(out, "fit_front.png"))
    overlay(r_best, a1, os.path.join(out, "fit_tq.png"))


def project_np(P3, c2w, fov, size):
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import project_points_to_image_batch
    pts, _, _ = project_points_to_image_batch(P3, torch.tensor(c2w, dtype=torch.float32, device=P3.device)[None],
                                              torch.tensor([fov], dtype=torch.float32, device=P3.device), size)
    return pts[0].double().cpu().numpy()


def cmd_landmarks(sv_dir, seed, out, mesh_pt=None):
    """the 3/4 camera from landmarks read in both views (landmarks.json), each front landmark lifted onto the
    single-view mesh (or mesh_pt, e.g. a two-view generation, to refine the camera on it) through the canonical
    front camera; then refined jointly with the silhouette. Silhouettes alone cannot tell a view from 10 deg above
    from one 10 deg below a head; landmarks can. Writes fit_landmarks.json with the silhouette-only, landmark-only
    and joint cameras, and fit.json = the joint one (the silhouette-only fit kept as fit_silhouette.json)"""
    import shutil
    from scipy.optimize import least_squares, minimize
    from frames import world_from_raw
    import views

    dev = "cuda"
    cam = json.load(open(os.path.join(sv_dir, "camera.json")))
    fov, d0 = cam["camera_angle_x"], cam["distance"]
    mesh_pt = mesh_pt or os.path.join(sv_dir, f"s{seed}.pt")
    print(f"[mesh] {mesh_pt}", flush=True)
    d = torch.load(mesh_pt, map_location="cpu")
    v = d["vertices"].double().numpy()
    coords, res = d["coords"].double().numpy(), d["res"]
    if coords.shape[1] == 4:
        coords = coords[:, 1:]
    world = torch.tensor(world_from_raw(v, coords, res)[0], dtype=torch.float32, device=dev)
    faces = d["faces"].long().to(dev)

    lm = json.load(open(os.path.join(out, "landmarks.json")))
    names = [k for k in lm["front"] if k in lm["tq"]]
    w = np.array([lm.get("weight", {}).get(k, 1.0) for k in names])
    S0 = Image.open(os.path.join(out, "view00_front.png")).size[0]
    S1 = Image.open(os.path.join(out, "view01_tq.png")).size[0]
    front_px = np.array([lm["front"][k] for k in names], float) + 0.5  # pixel index -> pixel centre
    target = np.array([lm["tq"][k] for k in names], float) + 0.5

    # lift: the front camera's z-buffer at the crop's own resolution
    views.project.__defaults__ = (fov,)  # views.py binds the Amina run's FOV as project()'s default
    c0 = torch.tensor(camera(0, 0, 0, d0, 0, 0), dtype=torch.float32, device=dev)
    fid, bary, _ = views.rasterize(world, faces, {"eye": c0[:3, 3], "x": c0[:3, 0], "y": c0[:3, 1], "z": c0[:3, 2]}, S0)
    P3 = []
    for k, (x, y) in zip(names, front_px.astype(int)):
        if int(fid[y, x]) < 0:  # a landmark on the outline (the ear) can fall a pixel outside the mesh's outline
            win = fid[y - 8:y + 9, x - 8:x + 9] >= 0
            assert win.any(), f"{k}: no surface within 8 px of the front landmark"
            dy, dx = (torch.nonzero(win).float() - 8).T
            i = int(torch.argmin(dx ** 2 + dy ** 2))
            print(f"[lift] {k}: off the mesh's outline, nearest surface pixel {float(np.hypot(dx[i].item(), dy[i].item())):.1f} px away", flush=True)
            x, y = x + int(dx[i]), y + int(dy[i])
        f = int(fid[y, x])
        P3.append((world[faces[f]] * bary[y, x][:, None]).sum(0))
    P3 = torch.stack(P3)
    back = np.abs(project_np(P3, camera(0, 0, 0, d0, 0, 0), fov, S0) - front_px).max()
    print(f"[lift] {len(names)} landmarks onto the mesh; reprojected through the front camera, max error {back:.2f} px", flush=True)
    del fid, bary
    torch.cuda.empty_cache()

    # Pixal3D's projection in float64 (its own runs in float32, too coarse for finite-difference Jacobians),
    # checked against it on these points
    P3n = P3.double().cpu().numpy()

    def proj64(c2w, size, fov_=fov):
        w2c = np.linalg.inv(c2w)
        pc = P3n @ w2c[:3, :3].T + w2c[:3, 3]
        f = (size / 2) / math.tan(fov_ / 2)
        return np.stack([f * pc[:, 0] / -pc[:, 2] + size / 2, -f * pc[:, 1] / -pc[:, 2] + size / 2], -1)

    chk = camera(-25, 5, 2, d0, 0.02, -0.01)
    dev64 = np.abs(proj64(chk, S1) - project_np(P3, chk, fov, S1)).max()
    print(f"[proj] float64 pinhole vs Pixal3D's projection on the landmarks: max diff {dev64:.4f} px", flush=True)
    assert dev64 < 0.05

    sw = np.sqrt(w)[:, None]
    resid = lambda x: ((proj64(camera(*x), S1) - target) * sw).ravel()

    def lm_err(x):
        e = np.linalg.norm(proj64(camera(*x), S1) - target, axis=1)
        return e, float(np.sqrt((w * e ** 2).sum() / w.sum()))

    sols = []
    for yaw0 in (-60, -45, -30, -15, 0):
        for pitch0 in (-20, -10, 0, 10, 20):
            r = least_squares(resid, np.array([yaw0, pitch0, 0, d0, 0, 0], float), loss="soft_l1", f_scale=5.0,
                              x_scale=np.array([1.0, 1.0, 1.0, 0.01, 0.005, 0.005]), diff_step=1e-6)
            sols.append((lm_err(r.x)[1], r.x, (yaw0, pitch0)))
    sols.sort(key=lambda t: t[0])
    distinct = []
    for s in sols:
        if all(abs(s[1][0] - t[1][0]) > 2 or abs(s[1][1] - t[1][1]) > 2 for t in distinct):
            distinct.append(s)
    for rms, x, st in distinct[:4]:
        print(f"[pnp] from yaw {st[0]:+d} pitch {st[1]:+d}: yaw {x[0]:+.2f} pitch {x[1]:+.2f} roll {x[2]:+.2f} d {x[3]:.4f} "
              f"u {x[4]:+.4f} v {x[5]:+.4f}: weighted RMS {rms:.2f} px", flush=True)
    x_lm = distinct[0][1]

    # the 3/4 photo is a separate image: does it want another lens? landmark fit per FOV, same starts
    fov_scan = {}
    for fdeg in (20.0, 26.0, math.degrees(fov), 40.0, 50.0):
        rf = lambda x, fr=math.radians(fdeg): ((proj64(camera(*x), S1, fr) - target) * sw).ravel()
        best = min((least_squares(rf, np.array([y0, p0, 0, d0 * math.tan(fov / 2) / math.tan(math.radians(fdeg) / 2), 0, 0]),
                                  loss="soft_l1", f_scale=5.0, x_scale=np.array([1.0, 1.0, 1.0, 0.01, 0.005, 0.005]), diff_step=1e-6)
                    for y0 in (-30, -15) for p0 in (-10, 10)), key=lambda r: r.cost)
        e = np.linalg.norm(proj64(camera(*best.x), S1, math.radians(fdeg)) - target, axis=1)
        fov_scan[f"{fdeg:.2f}"] = {"rms_px": float(np.sqrt((w * e ** 2).sum() / w.sum())), "yaw": best.x[0], "pitch": best.x[1]}
        print(f"[fov scan] {fdeg:5.2f} deg: landmark RMS {fov_scan[f'{fdeg:.2f}']['rms_px']:.2f} px, yaw {best.x[0]:+.2f} pitch {best.x[1]:+.2f}", flush=True)

    # silhouette term, as cmd_fit scores it
    world_s = world[torch.randperm(len(world), generator=torch.Generator().manual_seed(0))[:400_000].to(dev)]
    a1 = torch.tensor(np.asarray(Image.open(os.path.join(out, "view01_tq.png")).getchannel(3).resize((S, S), Image.Resampling.LANCZOS)) / 255.0,
                      dtype=torch.float32, device=dev)
    ys = torch.nonzero(a1.amax(1) > 0.5)[:, 0]
    rows = int(ys.min() + UPPER * (ys.max() - ys.min()))
    a1s = soft(a1)
    sil = lambda x: soft_iou(soft(render(world_s, camera(*x), fov)), a1s, rows)

    def joint(x):
        return (lm_err(x)[1] / 5.0) ** 2 + ((1 - sil(x)) / 0.01) ** 2

    x0 = x_lm
    simplex = np.array([x0] + [x0 + dlt for dlt in np.diag([2.0, 2.0, 1.5, 0.03 * x0[3], 0.01, 0.01])])
    r = minimize(joint, x0, method="Nelder-Mead", options={"initial_simplex": simplex, "xatol": 0.01, "fatol": 1e-4, "maxiter": 800})
    x_j = r.x

    src = os.path.join(out, "fit.json")
    if not os.path.exists(os.path.join(out, "fit_silhouette.json")):
        shutil.copy(src, os.path.join(out, "fit_silhouette.json"))
    x_s = np.array([json.load(open(os.path.join(out, "fit_silhouette.json")))[k] for k in ("yaw", "pitch", "roll", "distance", "u", "v")])

    report = {"landmarks": names, "weights": w.tolist(), "lift_max_err_px": float(back), "fov": fov, "d0": d0,
              "points_3d": dict(zip(names, P3n.tolist())), "fov_scan": fov_scan, "solutions": {}}
    for tag, x in (("silhouette", x_s), ("landmarks", x_lm), ("joint", x_j)):
        e, rms = lm_err(x)
        r_full = render(world, camera(*x), fov)
        rep = {"x": dict(zip(("yaw", "pitch", "roll", "distance", "u", "v"), map(float, x))), "landmark_rms_px": rms,
               "landmark_err_px": dict(zip(names, map(float, e))), "iou_upper": iou(r_full, a1, rows), "iou_full": iou(r_full, a1)}
        report["solutions"][tag] = rep
        print(f"[{tag:10s}] yaw {x[0]:+.2f} pitch {x[1]:+.2f} roll {x[2]:+.2f} d {x[3]:.4f} u {x[4]:+.4f} v {x[5]:+.4f}: "
              f"landmark RMS {rms:.1f} px (" + ", ".join(f"{k} {v_:.0f}" for k, v_ in zip(names, e)) + f"); "
              f"IoU above the neck {rep['iou_upper']:.3f}, whole {rep['iou_full']:.3f}", flush=True)
    json.dump(report, open(os.path.join(out, "fit_landmarks.json"), "w"), indent=2)
    j = report["solutions"]["joint"]
    fit = {"yaw": x_j[0], "pitch": x_j[1], "roll": x_j[2], "distance": x_j[3], "u": x_j[4], "v": x_j[5], "fov": fov, "d0": d0,
           "c2w": camera(*x_j).tolist(), "iou_upper": j["iou_upper"], "iou_full": j["iou_full"], "landmark_rms_px": j["landmark_rms_px"],
           "source": "joint landmarks + silhouette (fit_landmarks.json)"}
    json.dump(fit, open(src, "w"), indent=2, default=float)


def cmd_score(inp, gens):
    """for each generation (name=path.pt): the front landmarks lifted onto it through frame 0's camera and
    projected through frame 1's, against the 3/4 view's own landmarks (how far its shape agrees with the second
    photo, not only its outline), and the silhouette IoU in both views (whole, and above the 3/4 neck as the fit
    scores it). Writes <inp>/score.json"""
    from frames import world_from_raw
    import views

    dev = "cuda"
    meta = json.load(open(os.path.join(inp, "transforms.json")))
    fov = float(meta["camera_angle_x"])
    c2w = [np.array(f["transform_matrix"], float) for f in meta["frames"]]
    lm = json.load(open(os.path.join(inp, "landmarks.json")))
    names = [k for k in lm["front"] if k in lm["tq"]]
    w = np.array([lm.get("weight", {}).get(k, 1.0) for k in names])
    paths = [os.path.join(inp, f["file_path"]) for f in meta["frames"]]
    S0, S1 = Image.open(paths[0]).size[0], Image.open(paths[1]).size[0]
    front_px = np.array([lm["front"][k] for k in names], float) + 0.5
    target = np.array([lm["tq"][k] for k in names], float) + 0.5
    alphas = [torch.tensor(np.asarray(Image.open(p).getchannel(3).resize((S, S), Image.Resampling.LANCZOS)) / 255.0,
                           dtype=torch.float32, device=dev) for p in paths]
    ys = torch.nonzero(alphas[1].amax(1) > 0.5)[:, 0]
    rows = int(ys.min() + UPPER * (ys.max() - ys.min()))
    views.project.__defaults__ = (fov,)
    c0 = torch.tensor(c2w[0], dtype=torch.float32, device=dev)
    w2c1 = np.linalg.inv(c2w[1])
    f1 = (S1 / 2) / math.tan(fov / 2)
    score = {}
    for name, pt in gens:
        d = torch.load(pt, map_location="cpu")
        v = d["vertices"].double().numpy()
        coords, res = d["coords"].double().numpy(), d["res"]
        if coords.shape[1] == 4:
            coords = coords[:, 1:]
        world = torch.tensor(world_from_raw(v, coords, res)[0], dtype=torch.float32, device=dev)
        faces = d["faces"].long().to(dev)
        fid, bary, _ = views.rasterize(world, faces, {"eye": c0[:3, 3], "x": c0[:3, 0], "y": c0[:3, 1], "z": c0[:3, 2]}, S0)
        P3, moved = [], {}
        for k, (x, y) in zip(names, front_px.astype(int)):
            if int(fid[y, x]) < 0:
                win = fid[y - 8:y + 9, x - 8:x + 9] >= 0
                if not win.any():
                    P3.append(None)
                    continue
                dy, dx = (torch.nonzero(win).float() - 8).T
                i = int(torch.argmin(dx ** 2 + dy ** 2))
                moved[k] = float(np.hypot(dx[i].item(), dy[i].item()))
                x, y = x + int(dx[i]), y + int(dy[i])
            P3.append((world[faces[int(fid[y, x])]] * bary[y, x][:, None]).sum(0).double().cpu().numpy())
        del fid, bary
        ok = np.array([p is not None for p in P3])
        pc = np.array([p for p in P3 if p is not None]) @ w2c1[:3, :3].T + w2c1[:3, 3]
        proj = np.stack([f1 * pc[:, 0] / -pc[:, 2] + S1 / 2, -f1 * pc[:, 1] / -pc[:, 2] + S1 / 2], -1)
        e = np.linalg.norm(proj - target[ok], axis=1)
        rms = float(np.sqrt((w[ok] * e ** 2).sum() / w[ok].sum()))
        sil = [render(world, c, fov) for c in c2w]
        s = {"landmark_rms_px": rms, "landmark_err_px": dict(zip([n for n, o in zip(names, ok) if o], map(float, e))),
             "lifted_off_outline_px": moved, "missing": [n for n, o in zip(names, ok) if not o],
             "iou_front": iou(sil[0], alphas[0]), "iou_tq": iou(sil[1], alphas[1]), "iou_tq_upper": iou(sil[1], alphas[1], rows),
             "vertices": int(len(world))}
        score[name] = s
        print(f"[{name}] 3/4 landmark RMS {rms:.1f} px (" + ", ".join(f"{k} {x:.0f}" for k, x in s["landmark_err_px"].items()) +
              f"); silhouette IoU front {s['iou_front']:.4f}, 3/4 {s['iou_tq']:.4f} (above neck {s['iou_tq_upper']:.4f})", flush=True)
        del world, faces, sil
        torch.cuda.empty_cache()
    json.dump(score, open(os.path.join(inp, "score.json"), "w"), indent=2)


def cmd_build(sv_dir, out):
    from mv_prep import front
    fit = json.load(open(os.path.join(out, "fit.json")))
    cam = json.load(open(os.path.join(sv_dir, "camera.json")))
    assert abs(cam["distance"] - fit["d0"]) < 1e-9 and abs(cam["camera_angle_x"] - fit["fov"]) < 1e-12
    frames = [{"file_path": "view00_front.png", "name": "front", "transform_matrix": front(cam["distance"]).tolist()},
              {"file_path": "view01_tq.png", "name": "three_quarter", "transform_matrix": fit["c2w"]}]
    json.dump({"camera_angle_x": cam["camera_angle_x"], "mesh_scale": 1.0, "frames": frames},
              open(os.path.join(out, "transforms.json"), "w"), indent=2)
    print(f"[build] {len(frames)} frames -> {out}/transforms.json")


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "crop":
        cmd_crop(*args)
    elif cmd == "fit":
        cmd_fit(args[0], int(args[1]), args[2])
    elif cmd == "landmarks":
        cmd_landmarks(args[0], int(args[1]), args[2], *args[3:4])
    elif cmd == "score":
        cmd_score(args[0], [a.split("=", 1) for a in args[1:]])
    elif cmd == "build":
        cmd_build(*args)
    else:
        sys.exit(__doc__)
