"""Multi-view images of the Queen Amina statue: Qwen-Image-Edit-2511 + fal's Multiple-Angles LoRA,
run through the local ComfyUI API (127.0.0.1:8188).

    python3 gen_views.py <input.png> <out_dir> [--fast] [--seeds 1,2]

The graph is fal's published workflow (comfyui-workflow-multiple-angles.json): TextEncodeQwenImageEditPlus
with the input as reference, FluxKontextMultiReferenceLatentMethod "index_timestep_zero",
ModelSamplingAuraFlow 3.1, CFGNorm 1. Quality settings by default (no Lightning: 40 steps, true CFG 4);
--fast uses the 8-step Lightning LoRA at CFG 1 for previews.
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

# azimuth → the LoRA's descriptor (0° is the input's own view)
AZIMUTHS = {
    45: "front-right quarter view",
    90: "right side view",
    135: "back-right quarter view",
    180: "back view",
    225: "back-left quarter view",
    270: "left side view",
    315: "front-left quarter view",
}


def graph(image_name: str, prompt: str, seed: int, fast: bool, prefix: str) -> dict:
    g = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_edit_2511_fp8mixed.safetensors", "weight_dtype": "default"}},
        "angles": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["unet", 0], "lora_name": "qwen-image-edit-2511-multiple-angles-lora.safetensors", "strength_model": 0.9}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "load": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "scale": {"class_type": "FluxKontextImageScale", "inputs": {"image": ["load", 0]}},
        "pos": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["clip", 0], "vae": ["vae", 0], "image1": ["scale", 0], "prompt": prompt}},
        "neg": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["clip", 0], "vae": ["vae", 0], "image1": ["scale", 0], "prompt": ""}},
        "posm": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {"conditioning": ["pos", 0], "reference_latents_method": "index_timestep_zero"}},
        "negm": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {"conditioning": ["neg", 0], "reference_latents_method": "index_timestep_zero"}},
        "latent": {"class_type": "VAEEncode", "inputs": {"pixels": ["scale", 0], "vae": ["vae", 0]}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": prefix}},
    }
    model = ["angles", 0]
    if fast:
        g["lightning"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model, "lora_name": "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors", "strength_model": 1.0}}
        model = ["lightning", 0]
    g["shift"] = {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": model, "shift": 3.1}}
    g["norm"] = {"class_type": "CFGNorm", "inputs": {"model": ["shift", 0], "strength": 1.0}}
    g["sample"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": ["norm", 0], "positive": ["posm", 0], "negative": ["negm", 0], "latent_image": ["latent", 0],
            "seed": seed, "steps": 8 if fast else 40, "cfg": 1.0 if fast else 4.0,
            "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
        },
    }
    return g


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(API + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())


def wait(prompt_id: str) -> dict:
    while True:
        hist = json.loads(urllib.request.urlopen(f"{API}/history/{prompt_id}").read())
        if prompt_id in hist and hist[prompt_id].get("status", {}).get("completed"):
            return hist[prompt_id]
        if prompt_id in hist and hist[prompt_id].get("status", {}).get("status_str") == "error":
            raise RuntimeError(json.dumps(hist[prompt_id]["status"])[:2000])
        time.sleep(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("out")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--azimuths", default=",".join(map(str, AZIMUTHS)))
    ap.add_argument("--distance", default="wide shot", help="medium shot | wide shot | close-up")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"amina_mv_{uuid.uuid4().hex[:6]}.png"
    shutil.copy(a.image, COMFY_INPUT / name)
    client = uuid.uuid4().hex
    for az in map(int, a.azimuths.split(",")):
        for seed in map(int, a.seeds.split(",")):
            prompt = f"<sks> {AZIMUTHS[az]} eye-level shot {a.distance}"
            prefix = f"amina_mv/az{az:03d}_s{seed}{'_fast' if a.fast else ''}"
            t0 = time.time()
            pid = post("/prompt", {"prompt": graph(name, prompt, seed, a.fast, prefix), "client_id": client})["prompt_id"]
            h = wait(pid)
            img = h["outputs"]["save"]["images"][0]
            src = COMFY_OUTPUT / img["subfolder"] / img["filename"]
            dst = out / f"az{az:03d}_s{seed}{'_fast' if a.fast else ''}.png"
            shutil.copy(src, dst)
            print(f"{dst.name}  {prompt!r}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
