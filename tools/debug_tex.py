"""debug texture for the harness: visible from the allowed poses but seen by no view = white; seen = grey; never visible = dark"""
import sys, os, numpy as np, torch
sys.path.insert(0, "/tools")
from uvbake import uv_rasterize
from PIL import Image
mesh_p, fb_dir, bake_dir, out = sys.argv[1:5]
size = 2048
d = np.load(mesh_p)
fid, _ = uv_rasterize(torch.tensor(d["uvs"]), torch.tensor(d["faces"], dtype=torch.long), 4096)
tm = (fid >= 0).numpy()
needed = np.load(os.path.join(fb_dir, "needed.npy")); seen = np.load(os.path.join(bake_dir, "seen.npy"))
img = np.full((4096, 4096), 30, np.uint8)
img[tm] = np.where(needed & ~seen, 255, np.where(seen, 110, 30))
Image.fromarray(img).resize((size, size), Image.NEAREST).convert("RGB").save(out)
