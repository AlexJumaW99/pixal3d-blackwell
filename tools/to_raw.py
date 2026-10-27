"""mesh.npz (tools/webmesh.py) -> positions.f32, uvs.f32, faces.u32 for gltf/pack.mjs"""
import sys, numpy as np
d = np.load(sys.argv[1]); out = sys.argv[2]
d["positions"].astype("<f4").tofile(f"{out}/positions.f32")
d["uvs"].astype("<f4").tofile(f"{out}/uvs.f32")
d["faces"].astype("<u4").tofile(f"{out}/faces.u32")
print(d["positions"].shape, d["uvs"].shape, d["faces"].shape, "uv range", d["uvs"].min(0), d["uvs"].max(0))
