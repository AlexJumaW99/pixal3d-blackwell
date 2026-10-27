"""NAF feature upsampling in channel chunks, so Pixal3D's 1024 stages fit a 16 GB GPU.

NAF (valeoai/NAF, Apache-2.0) upsamples DINOv3's patch features with neighbourhood attention whose
queries and keys come from the guide image alone (256 channels, 4 heads); the features are only the
values. Each head's attention weights are therefore the same whichever of its value channels are
computed, so the 1024 value channels can be taken a slice of every head at a time, each slice
upsampled and projected into the voxel grid before the next. Pixal3D upsamples to 1024 x 1024 at
float32 at once (4 GiB a tensor, ~16 GiB in all), which the HF demo's 80 GB GPUs allow and a 16 GB
card does not.

check(): the chunked result against the original on a small target; tools/run_single.py patches
DinoV3ProjFeatureExtractor.forward with forward() below, which differs from Pixal3D's only in the
NAF branch.
"""
import torch
import torch.nn.functional as F

from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor


def naf_hr_projected(naf, image, lr_bchw, target_size, project, slice_per_head=64):
    """NAF(image, lr_bchw, target_size) projected by `project`, a slice of every head at a time.
    project: [B, C', H', W'] -> [B, G^3, C'], channel-wise (grid sampling)."""
    up = naf.upsampler
    heads = up.num_heads
    C = lr_bchw.shape[1]
    per_head = C // heads
    assert C % heads == 0 and per_head % slice_per_head == 0
    x = naf.image_encoder(image, output_size=target_size)
    q = naf.query_encoder(x)
    k = naf.key_encoder(x, lr_bchw)
    parts = []
    order = []
    for s in range(0, per_head, slice_per_head):
        idx = torch.cat([torch.arange(h * per_head + s, h * per_head + s + slice_per_head) for h in range(heads)]).to(lr_bchw.device)
        out = up(q, k, lr_bchw[:, idx], image)  # [B, heads * slice, H', W'], channels in idx order
        parts.append(project(out))
        order.append(idx)
        del out
        torch.cuda.empty_cache()
    z = torch.cat(parts, dim=-1)
    inv = torch.empty(C, dtype=torch.long, device=z.device)
    inv[torch.cat(order)] = torch.arange(C, device=z.device)
    return z[..., inv]


def forward(self, image, camera_angle_x=None, distance=None, mesh_scale=None, transform_matrix=None):
    """DinoV3ProjFeatureExtractor.forward (Pixal3D), the NAF branch through naf_hr_projected."""
    import numpy as np
    from PIL import Image
    if isinstance(image, torch.Tensor):
        assert image.ndim == 4
    elif isinstance(image, list):
        image = [i.resize((self.image_size, self.image_size), Image.LANCZOS) for i in image]
        image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
        image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
        image = torch.stack(image).cuda()
    else:
        raise ValueError(f"Unsupported type of image: {type(image)}")
    B = image.shape[0]
    if self.use_naf_upsample:
        image_for_naf = image.clone()
    image = self.transform(image)
    with torch.no_grad():
        z = self.extract_features(image)
        z_clstoken = z[:, 0:1]
        num_reg = getattr(self.model.config, 'num_register_tokens', 4)
        z_regtokens = z[:, 1:1 + num_reg]
        z_patchtokens = z[:, 1 + num_reg:]
        z_patchtokens_spatial = z_patchtokens.reshape(B, self.patch_number, self.patch_number, -1)
        if camera_angle_x is None or distance is None or mesh_scale is None:
            raise ValueError("camera_angle_x, distance, and mesh_scale must be provided")
        z_proj_lr = self.proj_grid(z_patchtokens_spatial, camera_angle_x, distance, mesh_scale, transform_matrix)
        if self.use_naf_upsample:
            self._load_naf()
            lr_features_bchw = z_patchtokens_spatial.permute(0, 3, 1, 2).contiguous()
            z_proj_hr = naf_hr_projected(
                self.naf_model, image_for_naf, lr_features_bchw, self.naf_target_size,
                lambda hr: self.proj_grid(hr, camera_angle_x, distance, mesh_scale, transform_matrix, BHWC=False),
            )
            z_proj = torch.cat([z_proj_lr, z_proj_hr], dim=-1)
            del z_proj_hr
        else:
            z_proj = z_proj_lr
        z_global = torch.cat([z_clstoken, z_regtokens], dim=1)
    return z_global, z_proj


def patch():
    DinoV3ProjFeatureExtractor.forward = forward


@torch.no_grad()
def check(naf, image, lr_bchw, target_size, project):
    """max |chunked - original| for one input (both through `project`)"""
    ref = project(naf(image, lr_bchw, target_size))
    got = naf_hr_projected(naf, image, lr_bchw, target_size, project)
    return (ref - got).abs().max().item(), ref.abs().max().item()
