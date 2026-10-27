import sys, torch
sys.path.insert(0, "/work"); sys.path.insert(0, "/tools")
from PIL import Image
import numpy as np
from naf_chunked import check
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor
m = DinoV3ProjFeatureExtractor(model_name="facebook/dinov3-vitl16-pretrain-lvd1689m", image_size=512, grid_resolution=32, use_naf_upsample=True, naf_target_size=512).eval().cuda()
m._load_naf()
img = Image.open("/out/sv/preprocessed.png").convert("RGB").resize((512, 512), Image.LANCZOS)
x = torch.from_numpy(np.asarray(img).astype(np.float32) / 255).permute(2, 0, 1)[None].cuda()
with torch.no_grad():
    z = m.extract_features(m.transform(x))
    p = z[:, 5:].reshape(1, m.patch_number, m.patch_number, -1).permute(0, 3, 1, 2)
    flat = lambda t: t.flatten(2).transpose(1, 2)
    for size in [(128, 128), (256, 256)]:
        d, mx = check(m.naf_model, x, p, size, flat)
        print(f"target {size}: max |chunked - original| = {d:.3e} (values up to {mx:.2f})")
