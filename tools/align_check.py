"""silhouette of a restyled view (statue on plain grey) against the clay render's alpha; best small shift"""
import sys, numpy as np, cv2
from PIL import Image
view, clay_alpha = sys.argv[1], sys.argv[2]
im = np.asarray(Image.open(view).convert("RGB")).astype(np.float32)
bg = np.median(np.concatenate([im[:20].reshape(-1, 3), im[-20:].reshape(-1, 3), im[:, :20].reshape(-1, 3), im[:, -20:].reshape(-1, 3)]), 0)
d = np.abs(im - bg).max(-1)
m = (d > 18).astype(np.uint8)
m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
n, lab, st, _ = cv2.connectedComponentsWithStats(m)
keep = np.zeros_like(m)
for i in range(1, n):
    if st[i, cv2.CC_STAT_AREA] > 400: keep[lab == i] = 1
keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
r = np.asarray(Image.open(clay_alpha)) > 127
k = keep.astype(bool)
iou = lambda a, b: (a & b).sum() / (a | b).sum()
best = (iou(k, r), 0, 0)
for dy in range(-8, 9):
    for dx in range(-8, 9):
        s = np.roll(np.roll(k, dy, 0), dx, 1)
        v = iou(s, r)
        if v > best[0]: best = (v, dx, dy)
print(f"IoU {iou(k, r):.4f}  best {best[0]:.4f} at dx={best[1]} dy={best[2]}  bg={bg.round(1)}")
Image.fromarray((keep * 255).astype(np.uint8)).save(view.replace(".png", "_mask.png"))
