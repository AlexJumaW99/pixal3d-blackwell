"""Download Pixal3D's multi-view weights (pipeline_mv.json and the ckpts/*_mv models) into HF_HOME (/hf)."""
import json
from huggingface_hub import hf_hub_download, snapshot_download

cfg = json.load(open(hf_hub_download("TencentARC/Pixal3D", "pipeline_mv.json")))
files = ["pipeline_mv.json"]
for v in cfg["args"]["models"].values():
    files += [f"{v}.json", f"{v}.safetensors"]
print("files", files, flush=True)
p = snapshot_download("TencentARC/Pixal3D", allow_patterns=files)
print("pixal3d mv", p, flush=True)
print("DONE", flush=True)
