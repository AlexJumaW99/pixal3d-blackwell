"""The front statue photo on the 3/4 view's dark backdrop, and both views as bust-only RGBA for Pixal3D.

    python3 tools/bg_swap.py <front.png> <three_quarter.png> <masks_dir> <out_dir>

masks_dir holds BiRefNet's masks from tools/birefnet_mask.py (<stem>_bn2048.png). Writes to out_dir:
  front_black.png         the front photo on the new backdrop; every pixel with alpha 1 is the original, verbatim
  front_rgba.png          statue + pedestal, straight alpha, edge colour with the old backdrop's spill removed
  front_bust_rgba.png     bust only, for Pixal3D (the two photos' pedestals are different objects)
  tq_bust_rgba.png        the 3/4 view's bust only, for Pixal3D
  matte.json              the fitted numbers the log prints
  debug/                  crops and overlays

Alpha, front photo (white marble on a light grey sweep, so colour alone cannot separate them):
- the backdrop B is a smooth studio sweep: a degree-4 polynomial per channel, fitted robustly to pixels well
  away from the subject (what is left is the grain);
- prior: BiRefNet's bust mask moved 0.7 px inward and narrowed to a 1.2 px ramp (its 0.5 contour sits ~0.7 px
  outside the true edge: measured from the rim profile, logged as [profile]), joined with the pedestal: its
  dark lower part by distance from B, connected to the bottom edge; the slab's polished top face, which mirrors
  the backdrop, as the polygon of lines fitted to its bevel highlight;
- within 2.5 px of that contour alpha is solved against the known backdrop, a = (I-B).(F-B)/|F-B|^2, F the rim
  colour 1.5-3.5 px inside, wherever |F-B| stands clear of the grain (a dead zone of 2.5 sigma/|F-B| keeps the
  grain from turning into a speckled halo); where it does not, the prior. Inside the band alpha is 1.
Colour at the edge: F = (I - (1-a)B)/a, the old backdrop's spill removed; the rim colour where a is small.
The new backdrop is the 3/4 view's own sweep (same fit) plus grain matched to its residual.
"""
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

GRAIN_SEED = 20261026


def load_rgb(p):
    return np.asarray(Image.open(p).convert("RGB")).astype(np.float32)


def load_mask(p):
    return np.asarray(Image.open(p)).astype(np.float32) / 65535


def poly_basis(h, w, deg):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    x = x / (w - 1) * 2 - 1
    y = y / (h - 1) * 2 - 1
    return np.stack([x ** i * y ** j for i in range(deg + 1) for j in range(deg + 1 - i)], -1)


def fit_backdrop(I, keep, deg=4, iters=3):
    """per-channel polynomial backdrop, refitted without the outliers of the previous fit"""
    A = poly_basis(*keep.shape, deg)
    used = keep.copy()
    for _ in range(iters):
        sub = np.argwhere(used)[::5]
        coef, *_ = np.linalg.lstsq(A[sub[:, 0], sub[:, 1]].astype(np.float64),
                                   I[sub[:, 0], sub[:, 1]].astype(np.float64), rcond=None)
        B = A @ coef.astype(np.float32)
        r = np.abs(I - B).max(-1)
        mad = np.median(r[used]) * 1.4826
        used = keep & (r < max(4 * mad, 6.0))
    res = (I - B)[used]
    return B, used, res.std(0)


def soft_sdf(m, eps=0.02):
    """first-order signed distance (px, + inside) to the 0.5 contour of a soft mask"""
    gy, gx = np.gradient(m)
    return (m - 0.5) / np.maximum(np.hypot(gx, gy), eps)


def sharpen_prior(m, shift, width=1.2):
    """the soft mask's 0.5 contour moved `shift` px inward, as a `width` px ramp"""
    return np.clip((soft_sdf(m) - shift) / width + 0.5, 0, 1).astype(np.float32)


def rim_profile(I, B, m, offsets=(-2, -1, 0, 1, 2, 3)):
    """(I-B) luma across the 0.5 contour of m, sampled along its normal; returns the sub-pixel offset at which
    half the rim contrast (offset +2..+3) is reached, and the per-offset medians"""
    gy, gx = np.gradient(cv2.GaussianBlur(m, (0, 0), 1.0))
    c = np.argwhere((m > 0.45) & (m < 0.55))
    n = np.stack([gy[c[:, 0], c[:, 1]], gx[c[:, 0], c[:, 1]]], -1)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    lum = (I - B).mean(-1)
    med = {}
    for o in offsets:
        p = np.rint(c + n * o).astype(int)
        p[:, 0] = p[:, 0].clip(0, m.shape[0] - 1)
        p[:, 1] = p[:, 1].clip(0, m.shape[1] - 1)
        med[o] = float(np.median(lum[p[:, 0], p[:, 1]]))
    rim = (med[2] + med[3]) / 2
    f = {o: med[o] / rim for o in offsets}
    edge = next(o - 1 + (0.5 - f[o - 1]) / (f[o] - f[o - 1]) for o in offsets[1:] if f[o - 1] < 0.5 <= f[o])
    return edge, med


def refine_alpha(I, B, prior, sigma_n, inner=2.5, outer=2.5, rim=(1.5, 3.5), cmin=8.0, cfull=18.0):
    m = prior > 0.5
    din = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dout = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    sd = np.where(m, din - 0.5, 0.5 - dout)
    band = (sd > -outer) & (sd < inner)
    src = (m & (sd >= rim[0]) & (sd <= rim[1])).astype(np.float32)
    w = cv2.GaussianBlur(src, (0, 0), 2.5)
    F_rim = np.stack([cv2.GaussianBlur(I[..., c] * src, (0, 0), 2.5) for c in range(3)], -1) / np.maximum(w, 1e-6)[..., None]
    D = F_rim - B
    c = np.sqrt((D * D).sum(-1))
    akb = ((I - B) * D).sum(-1) / np.maximum(c, 1e-3) ** 2
    t = np.clip(2.5 * sigma_n / np.maximum(c, 1e-3), 0, 0.35)
    akb = np.clip((akb - t) / (1 - 2 * t), 0, 1)
    rel = np.clip((c - cmin) / (cfull - cmin), 0, 1) * (w > 0.02)
    a = rel * akb + (1 - rel) * prior
    a = np.where(sd >= 1.0, np.maximum(a, prior), a)  # well inside: never more transparent than the prior
    a = np.where(band, a, (sd >= inner).astype(np.float32)).astype(np.float32)
    F_rim = np.where((w > 0.02)[..., None], F_rim, I)
    stats = {"band_px": int(band.sum()), "reliable_frac": float(rel[band & (sd < 1) & (sd > -1)].mean()),
             "contrast_median": float(np.median(c[band & (sd < 1) & (sd > -1)]))}
    return a, F_rim, stats


def foreground(I, B, a, F_rim):
    """straight-alpha colour: the old backdrop unmixed out, faded to the rim colour where alpha is small"""
    Fu = (I - (1 - a[..., None]) * B) / np.maximum(a, 1e-3)[..., None]
    s = np.clip((a - 0.1) / 0.3, 0, 1)[..., None]
    s = s * s * (3 - 2 * s)
    F = np.clip(s * Fu + (1 - s) * F_rim, 0, 255)
    return np.where((a >= 1.0)[..., None], I, F)


def scan_crossing(prof, thr):
    """first sub-pixel index at which prof rises through thr, or None"""
    above = np.nonzero(prof >= thr)[0]
    if len(above) == 0 or above[0] == 0:
        return None
    i = above[0]
    return i - 1 + (thr - prof[i - 1]) / (prof[i] - prof[i - 1])


def robust_line(u, v, iters=4, tol=1.0):
    """v = k u + c, refitted on inliers within tol px"""
    keep = np.ones(len(u), bool)
    for _ in range(iters):
        k, c = np.polyfit(u[keep], v[keep], 1)
        r = np.abs(v - (k * u + c))
        keep = r < max(tol, 2.5 * np.median(r[keep]))
    return k, c, int(keep.sum()), float(np.sqrt(np.mean(r[keep] ** 2)))


def slab_top(I, B, bust, ped, back=(1945, 2012), back_x=(375, 1418), sides_y=(1992, 2140),
             left_x=(170, 430), right_x=(1360, 1650), front_rows=(2170, 2225), thr=10.0):
    """the slab's top face as a polygon: back and side edges fitted to where its bevel highlight rises out of the
    backdrop (thr levels of (I-B) luma), sub-pixel; the front corners where the side lines meet the vertical ends
    of the slab's dark front face (ped, median over front_rows), with 12 px of overlap down into that face.
    Returns the polygon and the fits"""
    lum = cv2.blur((I - B).mean(-1), (3, 3))
    pts = []
    for x in range(*back_x):
        if bust[back[0]:back[1], x].max() > 0.02:
            continue
        y = scan_crossing(lum[back[0]:back[1], x], thr)
        if y is not None:
            pts.append((x, back[0] + y))
    pts = np.array(pts)
    kb, cb, nb, eb = robust_line(pts[:, 0], pts[:, 1])
    L, R = [], []
    for y in range(*sides_y):
        xl = scan_crossing(lum[y, left_x[0]:left_x[1]], thr)
        xr = scan_crossing(lum[y, right_x[0]:right_x[1]][::-1], thr)
        if xl is not None:
            L.append((y, left_x[0] + xl))
        if xr is not None:
            R.append((y, right_x[1] - 1 - xr))
    L, R = np.array(L), np.array(R)
    kl, cl, nl, el = robust_line(L[:, 0], L[:, 1])
    kr, cr, nr, er = robust_line(R[:, 0], R[:, 1])

    def meet(k, c):  # back: y = kb x + cb; side: x = k y + c
        y = (kb * c + cb) / (1 - kb * k)
        return k * y + c, y

    rows = ped[front_rows[0]:front_rows[1]]
    xl_end = float(np.median([np.nonzero(r)[0].min() for r in rows])) - 0.5  # pixel edge, not centre
    xr_end = float(np.median([np.nonzero(r)[0].max() for r in rows])) + 0.5
    yl, yr = (xl_end - cl) / kl, (xr_end - cr) / kr
    poly = np.array([meet(kl, cl), meet(kr, cr), (xr_end, yr), (xr_end, yr + 12), (xl_end, yl + 12), (xl_end, yl)])
    fits = {"back": {"k": kb, "c": cb, "inliers": nb, "of": len(pts), "rms_px": eb},
            "left": {"k": kl, "c": cl, "inliers": nl, "of": len(L), "rms_px": el},
            "right": {"k": kr, "c": cr, "inliers": nr, "of": len(R), "rms_px": er},
            "front_face_ends_x": [xl_end, xr_end], "polygon": poly.round(2).tolist()}
    return poly, fits


def render_polygon(poly, shape, ss=8):
    """anti-aliased polygon coverage; vertices in pixel-centre coordinates"""
    x0, y0 = np.floor(poly.min(0)).astype(int) - 2
    x1, y1 = np.ceil(poly.max(0)).astype(int) + 3
    canvas = np.zeros(((y1 - y0) * ss, (x1 - x0) * ss), np.uint8)
    p = np.rint((poly - [x0, y0] + 0.5) * ss - 0.5).astype(np.int32)
    cv2.fillPoly(canvas, [p], 255)
    cov = cv2.resize(canvas.astype(np.float32) / 255, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
    out = np.zeros(shape, np.float32)
    ys, xs = slice(max(y0, 0), min(y1, shape[0])), slice(max(x0, 0), min(x1, shape[1]))
    out[ys, xs] = cov[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0]
    return out


def pedestal_lower(I, B, bust, thr=14.0):
    """the dark pedestal: far from the backdrop model, not the bust, connected to the bottom edge"""
    d = np.abs(I - B).max(-1)
    cand = cv2.morphologyEx(((d > thr) & (bust < 0.5)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    lab, _ = ndimage.label(cand)
    ped = np.isin(lab, np.unique(lab[-1][lab[-1] > 0]))
    ped = cv2.morphologyEx(ped.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    return ndimage.binary_fill_holes(ped)


# 3/4 view: the chest plate's lower edge where it rests on the block, read off 2x zooms as (polarity, points).
# Left of the block's front corner (x ~705) the plate's shadowed front face meets the lit top of the block, a
# dark-to-bright step going down; right of it the lit plate ends in a crisp shadow line, bright-to-dark. The
# strongest step of either sign alone follows the plate's upper bevel on the left instead.
JUNCTION = [
    (+1, [(487, 1996), (540, 2008), (600, 2026), (650, 2045), (704, 2062)]),
    (-1, [(706, 2084), (800, 2086), (880, 2084), (960, 2086), (1040, 2084), (1100, 2074), (1150, 2059), (1196, 2040)]),
]


def bust_junction(I, segments=JUNCTION, corridor=8, deg=3):
    """per column, the strongest step of the segment's sign within `corridor` px of its read-off line, then a
    robust cubic per segment; returns the cut line y(x) for x across the image and the detections"""
    lum = cv2.blur(I.mean(-1), (5, 1))
    d = np.diff(cv2.GaussianBlur(lum, (0, 0), 1.0), axis=0)  # d[y] = L[y+1] - L[y]
    w = I.shape[1]
    xs = np.arange(w)
    y_cut = np.full(w, np.nan)
    det = []
    for sign, pts in segments:
        px, py = np.array(pts, float).T
        X = np.arange(int(px[0]), int(px[-1]) + 1)
        guide = np.interp(X, px, py)
        Y = np.array([int(g) - corridor + int(np.argmax(sign * d[int(g) - corridor:int(g) + corridor + 1, x])) + 0.5
                      for x, g in zip(X, guide)])
        keep = np.ones(len(X), bool)
        for _ in range(5):
            p = np.polyfit(X[keep], Y[keep], deg)
            r = np.abs(Y - np.polyval(p, X))
            keep = r < max(2.0, 3 * np.median(r[keep]))
        y_cut[X] = np.polyval(p, X)
        det.append((X, Y, keep, float(np.sqrt(np.mean(r[keep] ** 2))), float(np.abs(np.polyval(p, X) - guide).max())))
    known = ~np.isnan(y_cut)
    y_cut = np.interp(xs, xs[known], y_cut[known])  # flat beyond the ends, linear across the corner
    return y_cut, det


def grain_stats(I, B, keep):
    """std, lag-1/2 autocorrelation and channel correlation of the residual's fine part (above ~3 px)"""
    R = I - B
    hp = R - np.stack([cv2.GaussianBlur(R[..., c], (0, 0), 3.0) for c in range(3)], -1)
    k = cv2.erode(keep.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    std = hp[k].std(0)
    ac = {}
    for lag in (1, 2):
        a, b = hp[:, :-lag][k[:, :-lag] & k[:, lag:]], hp[:, lag:][k[:, :-lag] & k[:, lag:]]
        ac[lag] = float(np.mean([np.corrcoef(a[:, c], b[:, c])[0, 1] for c in range(3)]))
    cc = np.corrcoef(hp[k].T)
    return {"std": std.tolist(), "ac1": ac[1], "ac2": ac[2], "chan_corr": cc.tolist(), "hp": hp, "mask": k}


def make_grain(shape, st, seed=GRAIN_SEED):
    """Gaussian grain with the measured std, channel correlation and lag-1 autocorrelation: white noise through
    the separable 3-tap kernel [k, 1, k] whose lag-1 autocorrelation 2k / (1 + 2k^2) is the measured one (its
    lag-2, k^2 / (1 + 2k^2), then lands near the measured lag-2 too)"""
    rng = np.random.default_rng(seed)
    n = rng.standard_normal(shape[:2] + (3,)).astype(np.float32)
    rho = min(max(st["ac1"], 0.0), 0.7)
    k = (2 - np.sqrt(4 - 8 * rho * rho)) / (4 * rho) if rho > 1e-3 else 0.0
    if k > 0:
        kern = np.array([k, 1, k], np.float32)
        n = np.stack([cv2.sepFilter2D(n[..., c], -1, kern, kern, borderType=cv2.BORDER_REFLECT) for c in range(3)], -1)
        n /= n.reshape(-1, 3).std(0)
    L = np.linalg.cholesky(np.array(st["chan_corr"]))
    n = n @ L.T
    return n * np.array(st["std"], np.float32), float(k)


def save_rgb(path, x):
    Image.fromarray(np.clip(np.rint(x), 0, 255).astype(np.uint8)).save(path, compress_level=6)


def save_rgba(path, rgb, a):
    Image.fromarray(np.dstack([np.clip(np.rint(rgb), 0, 255), np.clip(np.rint(a * 255), 0, 255)]).astype(np.uint8), "RGBA").save(path, compress_level=6)


def zoom_sheet(path, tiles, boxes, z=4):
    rows = []
    for (y0, x0, h, w) in boxes:
        rows.append(np.concatenate([cv2.resize(t[y0:y0 + h, x0:x0 + w], (w * z, h * z), interpolation=cv2.INTER_NEAREST)
                                    for t in tiles], 1))
    save_rgb(path, np.concatenate(rows, 0))


def main(front, tq, masks, out):
    os.makedirs(os.path.join(out, "debug"), exist_ok=True)
    dbg = lambda n: os.path.join(out, "debug", n)
    log = {}
    I = load_rgb(front)
    J = load_rgb(tq)
    h, w = I.shape[:2]
    assert J.shape == I.shape, (I.shape, J.shape)
    sf, st = os.path.splitext(os.path.basename(front))[0], os.path.splitext(os.path.basename(tq))[0]
    bust = load_mask(os.path.join(masks, f"{sf}_bn2048.png"))
    tqm = load_mask(os.path.join(masks, f"{st}_bn2048.png"))

    # --- front backdrop: away from the bust, bright, not the pedestal zone or the watermark corner
    far = cv2.dilate((bust > 0.01).astype(np.uint8), np.ones((41, 41), np.uint8)) == 0
    keep = far & (I.mean(-1) > 195)
    keep[1940:, 160:1660] = False
    keep[-240:, -240:] = False
    keep = cv2.erode(keep.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    B, used, rstd = fit_backdrop(I, keep)
    sigma_n = float(rstd.mean())
    log["front_backdrop"] = {"fit_px": int(used.sum()), "residual_std": rstd.round(2).tolist()}
    print(f"[front backdrop] {used.sum()} px, residual std {rstd.round(2)}", flush=True)

    # --- where BiRefNet's contour sits relative to the true edge
    edge, med = rim_profile(I, B, bust)
    log["front_profile"] = {"edge_offset_px": edge, "median_I_minus_B": med}
    print(f"[profile] (I-B) luma across BiRefNet's 0.5 contour {dict((k, round(v, 1)) for k, v in med.items())}; "
          f"half the rim contrast at {edge:+.2f} px (inward)", flush=True)
    bust_p = sharpen_prior(bust, shift=edge)

    # --- pedestal
    ped = pedestal_lower(I, B, bust)
    poly, fits = slab_top(I, B, bust, ped)
    top = render_polygon(poly, (h, w))
    log["slab_top"] = fits
    for k in ("back", "left", "right"):
        f = fits[k]
        print(f"[slab {k:5s}] k {f['k']:+.4f} c {f['c']:.2f}: {f['inliers']}/{f['of']} inliers, rms {f['rms_px']:.2f} px", flush=True)
    print(f"[slab polygon] {np.round(poly, 1).tolist()}", flush=True)
    prior = np.maximum(np.maximum(bust_p, ped.astype(np.float32)), top)

    # --- alpha and colour
    a, F_rim, rs = refine_alpha(I, B, prior, sigma_n)
    F = foreground(I, B, a, F_rim)
    log["front_alpha"] = rs
    soft = (a > 0.02) & (a < 0.98)
    contour = cv2.morphologyEx((a > 0.5).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).sum() / 2
    log["front_alpha"]["edge_width_px"] = float(soft.sum() / contour)
    print(f"[front alpha] band {rs['band_px']} px, solved against the backdrop on {rs['reliable_frac']:.2f} of the contour "
          f"(median |F-B| {rs['contrast_median']:.1f}); mean edge width {soft.sum() / contour:.2f} px "
          f"(BiRefNet {((bust > 0.02) & (bust < 0.98)).sum() / (cv2.morphologyEx((bust > 0.5).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).sum() / 2):.2f})", flush=True)

    # --- the 3/4 view's backdrop and grain -> the new backdrop
    farj = cv2.dilate((tqm > 0.01).astype(np.uint8), np.ones((41, 41), np.uint8)) == 0
    farj[-240:, -240:] = False
    Bj, usedj, rstdj = fit_backdrop(J, farj)
    gs = grain_stats(J, Bj, usedj)
    grain, gsig = make_grain((h, w), gs)
    Bn = Bj + grain
    gchk = grain_stats(Bn, Bj, np.ones((h, w), bool))
    log["tq_backdrop"] = {"fit_px": int(usedj.sum()), "residual_std": rstdj.round(2).tolist(),
                          "grain": {k: gs[k] for k in ("std", "ac1", "ac2", "chan_corr")}, "grain_kernel_k": gsig,
                          "synth_grain": {k: gchk[k] for k in ("std", "ac1", "ac2")}}
    print(f"[3/4 backdrop] {usedj.sum()} px, residual std {rstdj.round(2)}; grain std {np.round(gs['std'], 2)} "
          f"ac1 {gs['ac1']:.3f} ac2 {gs['ac2']:.3f} -> synthetic std {np.round(gchk['std'], 2)} ac1 {gchk['ac1']:.3f} ac2 {gchk['ac2']:.3f}", flush=True)
    print(f"[3/4 backdrop] B at TL {Bj[0, 0].round(1)} TR {Bj[0, -1].round(1)} mid-L {Bj[h // 2, 0].round(1)} "
          f"mid-R {Bj[h // 2, -1].round(1)} BL {Bj[-1, 0].round(1)} BR {Bj[-1, -1].round(1)}", flush=True)

    # --- composite: alpha 1 is the original pixel, exactly
    O = a[..., None] * F + (1 - a[..., None]) * Bn
    O = np.where((a >= 1.0)[..., None], I, O)
    save_rgb(os.path.join(out, "front_black.png"), O)
    save_rgba(os.path.join(out, "front_rgba.png"), F, a)
    same = np.all(np.rint(O) == I, -1)
    log["front_black"] = {"pixels_identical_to_original": float(same.mean()), "alpha1_frac": float((a >= 1).mean())}
    print(f"[front_black] {same.mean():.4f} of all pixels identical to the original (alpha 1 on {(a >= 1).mean():.4f})", flush=True)

    # --- bust-only views for Pixal3D
    yb = fits["back"]["k"] * np.arange(w) + fits["back"]["c"]  # slab back edge per column
    below = np.arange(h)[:, None] >= (yb[None, :] - 4)
    a_fb = np.where(below, np.minimum(a, bust_p), a)
    a_fb = np.where(ped & ~(bust_p > 0), 0, a_fb).astype(np.float32)
    save_rgba(os.path.join(out, "front_bust_rgba.png"), F, a_fb)

    keepj = cv2.erode(usedj.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    edge_j, med_j = rim_profile(J, Bj, tqm)
    print(f"[3/4 profile] {dict((k, round(v, 1)) for k, v in med_j.items())}; half the rim contrast at {edge_j:+.2f} px", flush=True)
    a_j, F_rim_j, rs_j = refine_alpha(J, Bj, sharpen_prior(tqm, shift=edge_j), float(rstdj.mean()))
    F_j = foreground(J, Bj, a_j, F_rim_j)
    yj, det = bust_junction(J)
    for i, (X, Y, kj, rms, dev) in enumerate(det):
        print(f"[junction {i}] x {X.min()}..{X.max()}: {kj.sum()}/{len(X)} columns inliers, rms {rms:.2f} px, "
              f"max {dev:.1f} px from the read-off line", flush=True)
    xx = np.arange(w)
    cut = render_polygon(np.concatenate([np.stack([xx, yj], -1), [[w - 1, h + 2], [0, h + 2]]]), (h, w))
    a_jb = (a_j * (1 - cut)).astype(np.float32)
    lab, n = ndimage.label(a_jb > 0.5)
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
    main_c = 1 + int(np.argmax(sizes))
    a_jb = np.where((lab == main_c) | (lab == 0), a_jb, 0).astype(np.float32)
    save_rgba(os.path.join(out, "tq_bust_rgba.png"), F_j, a_jb)
    log["tq_alpha"] = {**rs_j, "edge_offset_px": edge_j, "dropped_islands": int(n - 1),
                       "junction": [{"x": [int(X.min()), int(X.max())], "inliers": int(kj.sum()), "of": len(X), "rms_px": rms,
                                     "max_dev_from_readoff_px": dev} for X, Y, kj, rms, dev in det]}
    print(f"[3/4 alpha] solved on {rs_j['reliable_frac']:.2f} of the contour (median |F-B| {rs_j['contrast_median']:.1f}); "
          f"{n - 1} islands dropped", flush=True)

    # --- debug views
    sm = lambda x: cv2.resize(x, (w // 3, h // 3), interpolation=cv2.INTER_AREA)
    a3 = lambda x: np.repeat(x[..., None] * 255, 3, -1)
    save_rgb(dbg("overview.png"), np.concatenate([sm(I), sm(O), sm(a3(a)), sm(F_j * a_jb[..., None]), sm(a3(a_jb))], 1))
    jv = J.copy()
    for X, Y, kj, _, _ in det:
        for x, y, k in zip(X, Y, kj):
            jv[int(y) - 3, x] = (255, 0, 0) if k else (255, 255, 0)
    for x in range(400, 1260):
        jv[int(round(yj[x])) + 3, x] = (0, 255, 0)
    save_rgb(dbg("tq_junction.png"), cv2.resize(jv[1940:2140, 440:1240], (1600, 400), interpolation=cv2.INTER_NEAREST))
    pv = I.copy()
    cv2.polylines(pv, [np.rint(poly).astype(np.int32)], True, (255, 0, 0), 1)
    save_rgb(dbg("slab_polygon.png"), pv[1930:2120, 150:1650])
    boxes = [(100, 760, 80, 80), (850, 320, 80, 80), (860, 1345, 80, 80), (1180, 380, 80, 80),
             (1940, 320, 80, 80), (1940, 1390, 80, 80), (2040, 180, 80, 80), (2040, 1540, 80, 80)]
    naive = I * bust[..., None] + (1 - bust[..., None]) * Bn
    zoom_sheet(dbg("front_edges.png"), [I, naive, O, a3(a)], boxes)
    json.dump(log, open(os.path.join(out, "matte.json"), "w"), indent=2)
    return I, B, a, F, Bn, O


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    main(*sys.argv[1:])
