"""GLB export for Pixal3D / TRELLIS.2 output without nvdiffrast (NVIDIA's licence is non-commercial).

o_voxel.postprocess.to_glb uses nvdiffrast only to rasterize triangles in UV space; everything
else (cumesh cleaning, remeshing, UV unwrapping, flex_gemm volume sampling) is MIT. This module
does the same steps with its own UV rasterizer (plain PyTorch on the GPU).

Texture convention, chosen so nothing has to be flipped: texel row r holds v = (r + 0.5) / size,
the image is saved with row 0 at the top, and the glTF uv is (u, v) as is (glTF puts v = 0 at the
top of the image).
"""
from typing import Dict

import cv2
import numpy as np
import torch
import trimesh
import trimesh.visual
from PIL import Image
import cumesh
from flex_gemm.ops.grid_sample import grid_sample_3d


@torch.no_grad()
def uv_rasterize(uvs: torch.Tensor, faces: torch.Tensor, size: int, chunk: int = 200_000):
    """For every texel centre inside a face's UV triangle: that face's id and barycentrics.
    Returns face_id (size, size) int64 (-1 where empty) and bary (size, size, 3)."""
    dev = uvs.device
    face_id = torch.full((size * size,), -1, dtype=torch.int64, device=dev)
    bary = torch.zeros((size * size, 3), dtype=torch.float32, device=dev)
    tri_all = uvs[faces.long()] * size  # (F, 3, 2) in texel units
    for f0 in range(0, faces.shape[0], chunk):
        tri = tri_all[f0 : f0 + chunk]
        lo = torch.floor(tri.min(1).values - 0.5).clamp(0, size - 1).long()
        hi = torch.ceil(tri.max(1).values - 0.5).clamp(0, size - 1).long()
        w = hi[:, 0] - lo[:, 0] + 1
        h = hi[:, 1] - lo[:, 1] + 1
        n = w * h
        fidx = torch.repeat_interleave(torch.arange(tri.shape[0], device=dev), n)
        start = torch.cumsum(n, 0) - n
        local = torch.arange(int(n.sum()), device=dev) - start[fidx]
        px = lo[fidx, 0] + local % w[fidx]
        py = lo[fidx, 1] + local // w[fidx]
        p = torch.stack([px.float() + 0.5, py.float() + 0.5], -1)
        a, b, c = tri[fidx, 0], tri[fidx, 1], tri[fidx, 2]
        v0, v1, v2 = b - a, c - a, p - a
        den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
        ok = den.abs() > 1e-12
        den = torch.where(ok, den, torch.ones_like(den))
        l1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
        l2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
        l0 = 1 - l1 - l2
        eps = -1e-6
        inside = ok & (l0 >= eps) & (l1 >= eps) & (l2 >= eps)
        pix = (py * size + px)[inside]
        face_id[pix] = fidx[inside] + f0
        bary[pix] = torch.stack([l0, l1, l2], -1)[inside]
    return face_id.view(size, size), bary.view(size, size, 3)


@torch.no_grad()
def export_glb(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    attr_volume: torch.Tensor,
    coords: torch.Tensor,
    attr_layout: Dict[str, slice],
    grid_size: int,
    aabb=((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)),
    decimation_target: int = 1_000_000,
    texture_size: int = 4096,
    remesh: bool = True,
    remesh_band: float = 1,
    remesh_project: float = 0.0,
) -> trimesh.Trimesh:
    """o_voxel.postprocess.to_glb's steps (same defaults as Pixal3D's inference.py), baked without nvdiffrast."""
    aabb = torch.tensor(aabb, dtype=torch.float32, device="cuda")
    gs = torch.tensor([grid_size] * 3, dtype=torch.int32, device="cuda")
    voxel_size = (aabb[1] - aabb[0]) / gs
    vertices, faces = vertices.cuda(), faces.cuda()

    mesh = cumesh.CuMesh()
    mesh.init(vertices, faces)
    mesh.fill_holes(max_hole_perimeter=3e-2)
    vertices, faces = mesh.read()
    bvh = cumesh.cuBVH(vertices, faces)
    if remesh:
        center = aabb.mean(dim=0)
        scale = (aabb[1] - aabb[0]).max().item()
        mesh.init(*cumesh.remeshing.remesh_narrow_band_dc(
            vertices, faces, center=center, scale=(grid_size + 3 * remesh_band) / grid_size * scale,
            resolution=grid_size, band=remesh_band, project_back=remesh_project, verbose=False, bvh=bvh,
        ))
        mesh.simplify(decimation_target, verbose=False)
    else:
        mesh.simplify(decimation_target * 3, verbose=False)
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=3e-2)
        mesh.simplify(decimation_target, verbose=False)
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=3e-2)
        mesh.unify_face_orientations()

    out_v, out_f, out_uv, out_vmaps = mesh.uv_unwrap(
        compute_charts_kwargs={"threshold_cone_half_angle_rad": np.radians(90.0), "refine_iterations": 0, "global_iterations": 1, "smooth_strength": 1},
        return_vmaps=True, verbose=False,
    )
    out_v, out_f, out_uv, out_vmaps = out_v.cuda(), out_f.cuda(), out_uv.cuda(), out_vmaps.cuda()
    mesh.compute_vertex_normals()
    out_n = mesh.read_vertex_normals()[out_vmaps]

    fid, bary = uv_rasterize(out_uv, out_f, texture_size)
    mask = fid >= 0
    tri = out_v[out_f[fid[mask]].long()]  # (N, 3, 3)
    pos = (tri * bary[mask].unsqueeze(-1)).sum(1)
    # attributes from the original surface, as to_glb does
    _, face_id, uvw = bvh.unsigned_distance(pos, return_uvw=True)
    pos = (vertices[faces[face_id.long()]] * uvw.unsqueeze(-1)).sum(1)
    attrs = torch.zeros(texture_size, texture_size, attr_volume.shape[1], device="cuda")
    attrs[mask] = grid_sample_3d(
        attr_volume, torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
        shape=torch.Size([1, attr_volume.shape[1], *gs.tolist()]),
        grid=((pos - aabb[0]) / voxel_size).reshape(1, -1, 3), mode="trilinear",
    )

    m = mask.cpu().numpy()
    a = attrs.cpu().numpy()
    def chan(name):
        return np.clip(a[..., attr_layout[name]] * 255, 0, 255).astype(np.uint8)
    inv = (~m).astype(np.uint8)
    base = cv2.inpaint(chan("base_color"), inv, 3, cv2.INPAINT_TELEA)
    metal = cv2.inpaint(chan("metallic"), inv, 1, cv2.INPAINT_TELEA)[..., None]
    rough = cv2.inpaint(chan("roughness"), inv, 1, cv2.INPAINT_TELEA)[..., None]
    alpha = cv2.inpaint(chan("alpha"), inv, 1, cv2.INPAINT_TELEA)[..., None]
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(np.concatenate([base, alpha], -1)),
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=Image.fromarray(np.concatenate([np.zeros_like(metal), rough, metal], -1)),
        metallicFactor=1.0, roughnessFactor=1.0, alphaMode="OPAQUE", doubleSided=not remesh,
    )
    v = out_v.cpu().numpy()
    n = out_n.cpu().numpy()
    uv = out_uv.cpu().numpy()
    # the same axis swap as to_glb (y, z = z, -y); the uv kept as is (see the module docstring)
    v[:, 1], v[:, 2] = v[:, 2], -v[:, 1].copy()
    n[:, 1], n[:, 2] = n[:, 2], -n[:, 1].copy()
    return trimesh.Trimesh(vertices=v, faces=out_f.cpu().numpy(), vertex_normals=n, process=False,
                           visual=trimesh.visual.TextureVisuals(uv=uv, material=material))
