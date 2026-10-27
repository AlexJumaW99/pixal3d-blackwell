"""Pixal3D multi-view generation for the MEROË Queen Amina statue.

    python /tools/run_mv.py --views /out/mv4/input --out /out/mv4/mv --seeds 7 [--num_views 4] [--resolution 1024]

<views> is Pixal3D's multi-view input: transforms.json plus one RGBA image per frame, frame 0 the canonical
front view (tools/mv_prep.py builds it). Differs from Pixal3D's inference_mv.py only where tools/run_single.py
does, for the same reasons:
- every view carries its own alpha, so the RMBG-2.0 background remover (CC BY-NC) is never loaded: its
  constructor is replaced by a stub that fails if it is ever called;
- DINOv3 comes from Meta's own gated repo (facebook/…), not the re-upload the script names;
- the GLB is baked by tools/uvbake.py instead of nvdiffrast (NVIDIA, non-commercial);
- NAF upsampling is done in channel slices (tools/naf_chunked.py). The multi-view extractor calls NAF from its
  own _project_single_view, which naf_chunked.patch() does not reach, so that method is patched here too.
For each seed it writes <out>/s<seed>.glb and the raw generation (<out>/s<seed>.pt).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, "/work")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] = "/out/flex_gemm_autotune.json"

import numpy as np
import torch

import pixal3d.pipelines.rembg as rembg_mod


class NoRembg:
    def __init__(self, *a, **k):
        pass

    def to(self, *a, **k):
        return self

    cuda = cpu = to

    def __call__(self, *a, **k):
        raise RuntimeError("background removal requested: every view must carry an alpha channel")


rembg_mod.BiRefNet = NoRembg

import inference_mv as inf  # Pixal3D's own inference_mv.py (view loading, model construction)

for cfg in inf.IMAGE_COND_CONFIGS.values():
    cfg["model_name"] = "facebook/dinov3-vitl16-pretrain-lvd1689m"

sys.path.insert(0, "/tools")
from uvbake import export_glb  # noqa: E402
import naf_chunked  # noqa: E402
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjMultiViewFeatureExtractor  # noqa: E402


def _project_single_view(self, z_patchtokens_spatial, image_for_naf, camera_angle_x_v, distance_v, mesh_scale_v, calc_mat_v):
    """DinoV3ProjMultiViewFeatureExtractor._project_single_view (Pixal3D), the NAF branch through naf_hr_projected."""
    z_proj_lr = self.proj_grid(z_patchtokens_spatial, camera_angle_x_v, distance_v, mesh_scale_v, calc_mat_v)
    if not self.use_naf_upsample:
        return z_proj_lr
    self._load_naf()
    lr_features_bchw = z_patchtokens_spatial.permute(0, 3, 1, 2).contiguous()
    z_proj_hr = naf_chunked.naf_hr_projected(
        self.naf_model, image_for_naf, lr_features_bchw, self.naf_target_size,
        lambda hr: self.proj_grid(hr, camera_angle_x_v, distance_v, mesh_scale_v, calc_mat_v, BHWC=False),
    )
    return torch.cat([z_proj_lr, z_proj_hr], dim=-1)


naf_chunked.patch()
DinoV3ProjMultiViewFeatureExtractor._project_single_view = _project_single_view

ROT = np.array([[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--num_views", type=int, default=None, help="use only the first N frames of transforms.json")
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--texture", type=int, default=4096)
    ap.add_argument("--faces", type=int, default=1_000_000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    t0 = time.time()
    pipeline = inf.init_pipeline(inf.MODEL_PATH, inf.CONFIG_FILE, low_vram=True)
    views = inf.load_views(a.views, a.num_views, rembg=None)
    inf.check_main_view(views)
    json.dump({"views": views["view_names"], "num_views": len(views["view_names"]), "resolution": a.resolution},
              open(os.path.join(a.out, "run.json"), "w"), indent=2)

    for seed in map(int, a.seeds.split(",")):
        t1 = time.time()
        torch.manual_seed(seed)
        meshes, (shape_slat, tex_slat, res) = pipeline.run_mv(
            views, seed=seed,
            sparse_structure_sampler_params={"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.7, "rescale_t": 5.0},
            shape_slat_sampler_params={"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.5, "rescale_t": 3.0},
            tex_slat_sampler_params={"steps": 12, "guidance_strength": 1.0, "guidance_rescale": 0.0, "rescale_t": 3.0},
            return_latent=True, pipeline_type=f"{a.resolution}_cascade", max_num_tokens=49152,
        )
        m = meshes[0]
        torch.save({"vertices": m.vertices.cpu(), "faces": m.faces.cpu(), "attrs": m.attrs.cpu(), "coords": m.coords.cpu(),
                    "res": res, "layout": {k: (v.start, v.stop) for k, v in pipeline.pbr_attr_layout.items()}},
                   os.path.join(a.out, f"s{seed}.pt"))
        print(f"[seed {seed}] generated in {time.time() - t1:.0f}s: {m.vertices.shape[0]} vertices, {m.faces.shape[0]} faces, grid {res}, "
              f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
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
