"""Pixal3D single-view generation for the MEROË Queen Amina statue.

    python /tools/run_single.py --image /mv/in/amina_rgba.png --out /out/sv --seeds 42,7 [--fov RAD] [--resolution 1536]

Differs from Pixal3D's inference.py only where licences require it:
- the input carries its own alpha (the site's matte), so the RMBG-2.0 background remover (CC BY-NC)
  is never loaded: its constructor is replaced by a stub that fails if it is ever called;
- DINOv3 comes from Meta's own gated repo (facebook/…), not the re-upload the script names;
- the GLB is baked by tools/uvbake.py instead of nvdiffrast (NVIDIA, non-commercial).
And where the 16 GB GPU requires it: NAF upsampling is done in channel slices (tools/naf_chunked.py).
For each seed it writes <out>/s<seed>.glb, the raw generation (<out>/s<seed>.pt) and the camera (<out>/camera.json).
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, "/work")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] = "/out/flex_gemm_autotune.json"

import numpy as np
import torch
from PIL import Image

import pixal3d.pipelines.rembg as rembg_mod


class NoRembg:
    def __init__(self, *a, **k):
        pass

    def to(self, *a, **k):
        return self

    cuda = cpu = to

    def __call__(self, *a, **k):
        raise RuntimeError("background removal requested: the input must carry an alpha channel")


rembg_mod.BiRefNet = NoRembg

import inference as inf  # Pixal3D's own inference.py (camera estimation, model construction)

for cfg in inf.IMAGE_COND_CONFIGS.values():
    cfg["model_name"] = "facebook/dinov3-vitl16-pretrain-lvd1689m"

sys.path.insert(0, "/tools")
from uvbake import export_glb  # noqa: E402
import naf_chunked  # noqa: E402

# NAF a slice of the channels at a time (identical result, a fraction of the memory: tools/naf_chunked.py)
naf_chunked.patch()

ROT = np.array([[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--fov", type=float, default=-1.0, help="horizontal FOV in radians; default: MoGe-2 estimate")
    ap.add_argument("--resolution", type=int, default=1536)
    ap.add_argument("--texture", type=int, default=4096)
    ap.add_argument("--faces", type=int, default=1_000_000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    t0 = time.time()
    pipeline = inf.init_pipeline(inf.MODEL_PATH, low_vram=True)
    img = Image.open(a.image)
    assert img.mode == "RGBA", "the input must be RGBA (its own matte)"
    pre = pipeline.preprocess_image(img)
    pre_path = os.path.join(a.out, "preprocessed.png")
    pre.save(pre_path)

    res_img, mesh_scale, extend = 512, 1.0, 0
    if a.fov > 0:
        d = inf.distance_from_fov(a.fov, torch.tensor([-1.0, 0.0, 0.0]), torch.tensor([0 - extend, res_img - 1 + extend]), mesh_scale, res_img)["distance_from_x"]
        camera = {"camera_angle_x": a.fov, "distance": d, "mesh_scale": mesh_scale, "source": "manual"}
    else:
        moge = inf.load_moge_model(device="cuda")
        camera = inf.get_camera_params_wild_moge(pre_path, moge, device="cuda", mesh_scale=mesh_scale, extend_pixel=extend, image_resolution=res_img)
        camera["source"] = "moge-2"
        moge.cpu()
        del moge
        torch.cuda.empty_cache()
    camera["fov_deg"] = math.degrees(camera["camera_angle_x"])
    print(f"[camera] {camera}", flush=True)
    json.dump(camera, open(os.path.join(a.out, "camera.json"), "w"), indent=2)

    for seed in map(int, a.seeds.split(",")):
        t1 = time.time()
        torch.manual_seed(seed)
        meshes, (shape_slat, tex_slat, res) = pipeline.run(
            pre, camera_params=camera, seed=seed,
            sparse_structure_sampler_params={"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.7, "rescale_t": 5.0},
            shape_slat_sampler_params={"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.5, "rescale_t": 3.0},
            tex_slat_sampler_params={"steps": 12, "guidance_strength": 1.0, "guidance_rescale": 0.0, "rescale_t": 3.0},
            preprocess_image=False, return_latent=True, pipeline_type=f"{a.resolution}_cascade", max_num_tokens=49152,
        )
        m = meshes[0]
        torch.save({"vertices": m.vertices.cpu(), "faces": m.faces.cpu(), "attrs": m.attrs.cpu(), "coords": m.coords.cpu(),
                    "res": res, "layout": {k: (v.start, v.stop) for k, v in pipeline.pbr_attr_layout.items()}},
                   os.path.join(a.out, f"s{seed}.pt"))
        print(f"[seed {seed}] generated in {time.time() - t1:.0f}s: {m.vertices.shape[0]} vertices, {m.faces.shape[0]} faces, grid {res}", flush=True)
        torch.cuda.empty_cache()
        glb = export_glb(m.vertices, m.faces, m.attrs, m.coords, pipeline.pbr_attr_layout, grid_size=res,
                         decimation_target=a.faces, texture_size=a.texture)
        glb.apply_transform(ROT)
        glb.export(os.path.join(a.out, f"s{seed}.glb"), extension_webp=True)
        print(f"[seed {seed}] GLB in {time.time() - t1:.0f}s total", flush=True)
        del meshes, m, glb
        torch.cuda.empty_cache()
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
