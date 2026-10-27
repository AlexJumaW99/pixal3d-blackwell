"""Download Pixal3D's single-view weights, MoGe-2 and DINOv3 into HF_HOME (/hf)."""
import json
from huggingface_hub import hf_hub_download, snapshot_download

cfg = json.load(open(hf_hub_download("TencentARC/Pixal3D", "pipeline.json")))
files = ["pipeline.json", "LICENSE", "NOTICE"]
for v in cfg["args"]["models"].values():
    files += [f"{v}.json", f"{v}.safetensors"]
p = snapshot_download("TencentARC/Pixal3D", allow_patterns=files)
print("pixal3d", p, flush=True)
print("moge", snapshot_download("Ruicheng/moge-2-vitl"), flush=True)
print("dinov3", snapshot_download("facebook/dinov3-vitl16-pretrain-lvd1689m"), flush=True)
print("DONE", flush=True)
