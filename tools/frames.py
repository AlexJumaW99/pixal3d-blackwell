"""Frames of a Pixal3D generation, found from the geometry rather than from bounding extents.

tools/mv_prep.py's match_axes pairs axes by their min/max extents. That cannot tell a sign when an axis's extent is
symmetric about 0 (a head centred in depth): on 2026-10-26 it mirrored a two-view bust front to back. Here every
signed axis permutation (all 48) is scored on the geometry itself:
- raw_to_index: raw vertices -> voxel-index frame, by the fraction of vertices that land in an occupied voxel
  (or a face neighbour of one: a vertex on a voxel face can round either way);
- glb_to_raw: an exported GLB's vertices -> raw frame, by the mean distance to the nearest raw vertex.
Both print the best and the runner-up so an ambiguity shows, and refuse a weak or narrow winner.
"""
import itertools

import numpy as np
from scipy.spatial import cKDTree

GRID_R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)  # ProjGrid: grid index frame -> projection frame


def signed_perms():
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3))
            for r, (c, s) in enumerate(zip(perm, signs)):
                M[r, c] = s
            yield M


def raw_to_index(v, coords, res, n=60_000, seed=0):
    """P (3x3 signed permutation) with voxel position ~ P @ v, v the raw vertices in [-0.5, 0.5]"""
    coords = np.asarray(coords)
    if coords.shape[1] == 4:
        coords = coords[:, 1:]
    R = res + 2  # keys of voxel (x, y, z) shifted by 1 so face neighbours never wrap
    key = lambda c: ((c[:, 0] + 1) * R + (c[:, 1] + 1)) * R + (c[:, 2] + 1)
    c = coords.astype(np.int64)
    nb = [c] + [c + d for d in np.vstack([np.eye(3, dtype=np.int64), -np.eye(3, dtype=np.int64)])]
    near = np.unique(np.concatenate([key(x) for x in nb]))
    rng = np.random.default_rng(seed)
    vs = np.asarray(v, np.float64)[rng.choice(len(v), min(n, len(v)), replace=False)]
    scores = []
    for M in signed_perms():
        k = key(np.floor((vs @ M.T + 0.5) * res).astype(np.int64))
        i = np.clip(np.searchsorted(near, k), 0, len(near) - 1)
        scores.append((float((near[i] == k).mean()), M))
    scores.sort(key=lambda t: -t[0])
    (b, P), (r2, _) = scores[0], scores[1]
    print(f"[frame raw->index] best {b:.4f} (det {np.linalg.det(P):+.0f}), runner-up {r2:.4f}", flush=True)
    assert b > 0.9 and b - r2 > 0.1, f"raw->index frame not determined: {b:.3f} vs {r2:.3f}"
    return P


def glb_to_raw(g, v, n=20_000, seed=0):
    """Q (3x3 signed permutation, a rotation) with raw vertex ~ Q @ g, g an exported GLB's vertices. Only rotations
    compete: the export (run_mv's ROT, uvbake's y/z swap) never mirrors, and a bust's left-right mirror lies almost on
    its own surface, so letting mirrors compete would only blur the decision; the best mirror is printed anyway"""
    rng = np.random.default_rng(seed)
    v = np.asarray(v, np.float64)
    tree = cKDTree(v[rng.choice(len(v), min(400_000, len(v)), replace=False)])
    gs = np.asarray(g, np.float64)[rng.choice(len(g), min(n, len(g)), replace=False)]
    scores = sorted(((float(tree.query(gs @ M.T)[0].mean()), M) for M in signed_perms()), key=lambda t: t[0])
    rot = [s for s in scores if np.linalg.det(s[1]) > 0]
    mir = [s for s in scores if np.linalg.det(s[1]) < 0]
    (b, Q), (r2, _) = rot[0], rot[1]
    print(f"[frame glb->raw] best rotation: mean distance {b:.5f}; runner-up rotation {r2:.5f}; best mirror {mir[0][0]:.5f}", flush=True)
    assert b < 0.005 and r2 > 3 * b, f"glb->raw frame not determined: {b:.5f} vs {r2:.5f}"
    return Q


def world_from_raw(v, coords, res):
    """raw vertices in the projection frame, and the 3x3 that took them there"""
    M = GRID_R @ raw_to_index(v, coords, res)
    return np.asarray(v, np.float64) @ M.T, M
