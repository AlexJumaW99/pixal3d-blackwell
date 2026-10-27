"""BiRefNet (ZhengPeng7/BiRefNet, MIT) foreground masks on the CPU, for the statue background swap.

    python /tools/birefnet_mask.py <model_dir> <out_dir> <image>... [--sizes 1024,2048]

For each image and size: the image is resized to size x size (as BiRefNet's own inference does), the sigmoid
mask is resized back to the image's own size (bilinear, in float) and saved as <out_dir>/<stem>_bn<size>.png,
16-bit grey. Not RMBG-2.0 (CC BY-NC), which is why the Pixal3D runs never load the pipeline's own remover.
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageSegmentation

MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out_dir")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--sizes", default="1024,2048")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    torch.set_num_threads(os.cpu_count())

    t0 = time.time()
    model = AutoModelForImageSegmentation.from_pretrained(a.model_dir, trust_remote_code=True).float().eval()
    print(f"[model] loaded in {time.time() - t0:.0f}s", flush=True)

    for path in a.images:
        img = Image.open(path).convert("RGB")
        x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255
        stem = os.path.splitext(os.path.basename(path))[0]
        for size in map(int, a.sizes.split(",")):
            t1 = time.time()
            inp = F.interpolate(x[None], (size, size), mode="bilinear", align_corners=False, antialias=True)
            with torch.no_grad():
                pred = model((inp - MEAN) / STD)[-1].sigmoid()
            mask = F.interpolate(pred, (img.height, img.width), mode="bilinear", align_corners=False)[0, 0].clamp(0, 1)
            m = mask.numpy()
            Image.fromarray((m * 65535).round().astype(np.uint16)).save(os.path.join(a.out_dir, f"{stem}_bn{size}.png"))
            print(f"[{stem} @{size}] {time.time() - t1:.0f}s: mask > 0.5 covers {(m > 0.5).mean():.4f}, "
                  f"soft px (0.02..0.98) {((m > 0.02) & (m < 0.98)).mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
