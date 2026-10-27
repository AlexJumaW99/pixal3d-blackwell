"""A web-weight copy of a Pixal3D GLB: welded, simplified with cumesh, UV-unwrapped, in the GLB's own frame.

    python /tools/webmesh.py <in.glb> <out.npz> [faces=300000]

Writes positions, normals, faces and uvs (no texture: tools/bake.py makes it), and prints the counts.
"""
import sys

import numpy as np
import torch
import trimesh
import cumesh

src, out = sys.argv[1], sys.argv[2]
target = int(sys.argv[3]) if len(sys.argv) > 3 else 300_000
scene = trimesh.load(src, force="scene")
mesh = trimesh.util.concatenate([g for g in scene.dump()])
mesh.merge_vertices(merge_tex=True, merge_norm=True)  # weld the UV seams: geometry only
print("in", len(mesh.vertices), "vertices", len(mesh.faces), "faces")
cm = cumesh.CuMesh()
cm.init(torch.tensor(mesh.vertices, dtype=torch.float32, device="cuda"), torch.tensor(mesh.faces, dtype=torch.int32, device="cuda"))
cm.remove_duplicate_faces()
cm.repair_non_manifold_edges()
cm.remove_small_connected_components(1e-5)
cm.fill_holes(max_hole_perimeter=3e-2)
cm.simplify(target, verbose=False)
cm.remove_duplicate_faces()
cm.repair_non_manifold_edges()
cm.unify_face_orientations()
v, f, uv, vmaps = cm.uv_unwrap(
    compute_charts_kwargs={"threshold_cone_half_angle_rad": np.radians(90.0), "refine_iterations": 0, "global_iterations": 1, "smooth_strength": 1},
    return_vmaps=True, verbose=False,
)
cm.compute_vertex_normals()
n = cm.read_vertex_normals()[vmaps.cuda()]
np.savez(out, positions=v.cpu().numpy().astype(np.float32), normals=n.cpu().numpy().astype(np.float32),
         faces=f.cpu().numpy().astype(np.int32), uvs=uv.cpu().numpy().astype(np.float32))
print("out", v.shape[0], "vertices", f.shape[0], "faces")
