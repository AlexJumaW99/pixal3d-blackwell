"""Restyle geometry renders of the Amina statue to look like the original, with Qwen-Image-Edit-2511 in ComfyUI.

    python3 restyle.py <views_dir> <appearance.png> <out_dir> [--seeds 1] [--names yaw045,...]

Image 1 is a clay render of the 3D model from a known camera (tools/views.py): it fixes every outline.
Image 2 is the original statue: it gives the material, the lighting and the detail. No camera LoRA: the
camera is the render's. 40 steps, true CFG 4 (the model's own quality settings).
"""
import argparse
import json
import shutil
import time
import urllib.request
import uuid
from pathlib import Path

API = "http://127.0.0.1:8188"
COMFY_INPUT = Path("/home/meroe/docker-stacks/comfy-ui/basedir/input")
COMFY_OUTPUT = Path("/home/meroe/docker-stacks/comfy-ui/basedir/output")

PROMPT = (
    "Make the sculpture in image 1 the same dark cast-metal equestrian statue as in image 2: the same dark, "
    "polished, speckled and mottled metal on the horse, the same quilted armour, veil, turban, cape, bridle and "
    "saddle cloth, the same black-and-white studio photograph lighting, on the same plain grey background. "
    "Keep the camera angle, the pose, the outline and every shape of image 1 exactly as they are."
)


def graph(structure: str, appearance: str, seed: int, prefix: str, denoise: float = 1.0) -> dict:
    return {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_edit_2511_fp8mixed.safetensors", "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "s": {"class_type": "LoadImage", "inputs": {"image": structure}},
        "a": {"class_type": "LoadImage", "inputs": {"image": appearance}},
        "s1": {"class_type": "FluxKontextImageScale", "inputs": {"image": ["s", 0]}},
        "a1": {"class_type": "FluxKontextImageScale", "inputs": {"image": ["a", 0]}},
        "pos": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["clip", 0], "vae": ["vae", 0], "image1": ["s1", 0], "image2": ["a1", 0], "prompt": PROMPT}},
        "neg": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["clip", 0], "vae": ["vae", 0], "image1": ["s1", 0], "image2": ["a1", 0], "prompt": ""}},
        "posm": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {"conditioning": ["pos", 0], "reference_latents_method": "index_timestep_zero"}},
        "negm": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {"conditioning": ["neg", 0], "reference_latents_method": "index_timestep_zero"}},
        "latent": {"class_type": "VAEEncode", "inputs": {"pixels": ["s1", 0], "vae": ["vae", 0]}},
        "shift": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["unet", 0], "shift": 3.1}},
        "norm": {"class_type": "CFGNorm", "inputs": {"model": ["shift", 0], "strength": 1.0}},
        "sample": {"class_type": "KSampler", "inputs": {"model": ["norm", 0], "positive": ["posm", 0], "negative": ["negm", 0], "latent_image": ["latent", 0],
                                                        "seed": seed, "steps": 40, "cfg": 4.0, "sampler_name": "euler", "scheduler": "simple", "denoise": denoise}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": prefix}},
    }


def post(path, body):
    req = urllib.request.Request(API + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())


def wait(pid):
    while True:
        h = json.loads(urllib.request.urlopen(f"{API}/history/{pid}").read())
        if pid in h and h[pid].get("status", {}).get("completed"):
            return h[pid]
        if pid in h and h[pid].get("status", {}).get("status_str") == "error":
            raise RuntimeError(json.dumps(h[pid]["status"])[:2000])
        time.sleep(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("views")
    ap.add_argument("appearance")
    ap.add_argument("out")
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--names", default="yaw045,yaw090,yaw135,yaw180,yaw225,yaw270,yaw315")
    ap.add_argument("--denoise", type=float, default=1.0, help="below 1: start from the render (img2img), which locks the composition")
    ap.add_argument("--tint", action="store_true", help="darken the clay towards the statue's tones first")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = uuid.uuid4().hex[:6]
    app = f"amina_app_{tag}.png"
    shutil.copy(a.appearance, COMFY_INPUT / app)
    for name in a.names.split(","):
        st = f"amina_{name}_{tag}.png"
        if a.tint:
            import numpy as np
            from PIL import Image
            clay = np.asarray(Image.open(Path(a.views) / f"{name}_clay.png").convert("RGB")).astype(np.float32)
            alpha = np.asarray(Image.open(Path(a.views) / f"{name}_alpha.png")) > 127
            dark = (clay / 255) ** 1.6 * 150  # the statue is dark metal: mean ~70 against the clay's ~150
            Image.fromarray(np.where(alpha[..., None], dark, clay).clip(0, 255).astype(np.uint8)).save(COMFY_INPUT / st)
        else:
            shutil.copy(Path(a.views) / f"{name}_clay.png", COMFY_INPUT / st)
        for seed in map(int, a.seeds.split(",")):
            t0 = time.time()
            pid = post("/prompt", {"prompt": graph(st, app, seed, f"amina_restyle/{name}_s{seed}", a.denoise), "client_id": tag})["prompt_id"]
            img = wait(pid)["outputs"]["save"]["images"][0]
            dst = out / f"{name}_s{seed}{'' if a.denoise >= 1 else f'_d{round(a.denoise * 100)}'}.png"
            shutil.copy(COMFY_OUTPUT / img["subfolder"] / img["filename"], dst)
            print(f"{dst.name} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
