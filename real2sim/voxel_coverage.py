#!/usr/bin/env python3
"""STO-SCN-103 — voxel-coverage best-N view selector (coverage-optimization greedy).

Replaces STO-SCN-094's track-covisibility objective (which over-rewarded clustered
same-angle cameras — the operator's "95% same coverage from the same angle") with a
SURFACE-COVERAGE objective: voxelize the scene, reward each camera for the voxel faces
it observes weighted by incidence-angle FLUX (90deg hit = 1.0, grazing -> 0), and
greedy-add the camera with the largest MARGINAL coverage gain. A redundant same-angle
camera adds ~0; a complementary angle on the same surface is rewarded -> real variety.

Published formulation (not a local invention): "Coverage Optimization for Camera View
Selection" (arXiv 2604.05259, 2026); "Efficient View Clustering and Selection for
City-Scale 3D Reconstruction" (arXiv 2207.08434).

Gauge-free (voxel size = fraction of the scene-bbox diagonal; the FastMap/DA3 gauge is
non-metric until scale calibration, STO-SCN-016). Deterministic, pure-CPU (numpy).

First-light scope (STO-SCN-103): frustum + incidence only. OCCLUSION (voxel-grid
ray-march line-of-sight) is the next increment — see the story § Out of scope.

Usage:
  voxel_coverage.py <sparse_dir> --n 24 [--grid 64] [--out selection.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import covis_graph as cg            # noqa: E402  (read_points3D_bin)
import posed_from_sparse as pfs     # noqa: E402  (readers)

_NEIGHBORS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def voxelize(points, grid=64):
    """points (N,3) -> (origin(3,), vsize float, occupied set[(i,j,k)], diag float).
    Voxel size = bbox-diagonal / grid (gauge-independent)."""
    pts = np.asarray(points, dtype=np.float64)
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    diag = float(np.linalg.norm(mx - mn))
    vsize = diag / grid if diag > 0 else 1.0
    ijk = np.floor((pts - mn) / vsize).astype(np.int64)
    occupied = set(map(tuple, ijk.tolist()))
    return mn, vsize, occupied, diag


def exposed_faces(origin, vsize, occupied):
    """-> (centers (F,3), normals (F,3)). A face is exposed when the neighbor voxel
    across it is empty (it borders free space = a camera could observe it)."""
    origin = np.asarray(origin, dtype=np.float64)
    centers, normals = [], []
    for (i, j, k) in occupied:
        vc = origin + (np.array([i, j, k]) + 0.5) * vsize
        for (di, dj, dk) in _NEIGHBORS:
            if (i + di, j + dj, k + dk) in occupied:
                continue                                  # interior face — skip
            nrm = np.array([di, dj, dk], dtype=np.float64)
            centers.append(vc + 0.5 * vsize * nrm)
            normals.append(nrm)
    if not centers:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.asarray(centers), np.asarray(normals)


def _wc_parts(w2c):
    """w2c(4x4) -> (R_w2c (3,3), camera center (3,))."""
    R = np.asarray([[w2c[r][c] for c in range(3)] for r in range(3)], dtype=np.float64)
    t = np.asarray([w2c[r][3] for r in range(3)], dtype=np.float64)
    center = -R.T @ t
    return R, center


def camera_weights(face_c, face_n, w2c, intr, near, far):
    """Per-(camera,face) coverage weight: max(0, cos incidence) for faces inside the
    frustum (in front, within FOV, within depth), else 0. Vectorised over faces."""
    R, center = _wc_parts(w2c)
    t = -R @ center
    Xc = face_c @ R.T + t                                 # world -> camera (F,3)
    z = Xc[:, 2]
    infront = z > near
    zc = np.where(infront, z, 1.0)
    u = intr["fx"] * Xc[:, 0] / zc + intr["cx"]
    v = intr["fy"] * Xc[:, 1] / zc + intr["cy"]
    in_fov = infront & (z < far) & (u >= 0) & (u < intr["w"]) & (v >= 0) & (v < intr["h"])
    ray = center - face_c                                 # face -> camera (F,3)
    rn = ray / (np.linalg.norm(ray, axis=1, keepdims=True) + 1e-12)
    cos = np.sum(face_n * rn, axis=1)                     # 1 at 90deg hit, <=0 = behind
    return np.where(in_fov, np.maximum(0.0, cos), 0.0)


def coverage_matrix(face_c, face_n, cameras, near, far):
    """-> W (C,F): camera c's flux weight on each exposed face."""
    W = np.zeros((len(cameras), len(face_c)))
    for c, cam in enumerate(cameras):
        W[c] = camera_weights(face_c, face_n, cam["w2c"], cam["intr"], near, far)
    return W


def greedy_select(W, n):
    """Greedy submodular coverage maximisation. Each step add the camera with the
    largest MARGINAL gain (sum of per-face improvement over the current best coverage).
    Returns (order[list of cam idx], face_cov(F,), per_step_gain[list])."""
    C, F = W.shape
    cov = np.zeros(F)
    order, gains, chosen = [], [], set()
    for _ in range(min(n, C)):
        marginal = np.maximum(0.0, W - cov).sum(axis=1)   # (C,)
        if chosen:
            marginal[list(chosen)] = -1.0
        c = int(np.argmax(marginal))                      # deterministic: first max
        if marginal[c] <= 1e-9:
            break                                         # coverage saturated
        order.append(c); chosen.add(c); gains.append(float(marginal[c]))
        cov = np.maximum(cov, W[c])
    return order, cov, gains


def _view_spread(cameras, order):
    """Median pairwise optical-axis angle (deg) of the selected cameras."""
    fwds = []
    for c in order:
        R, _ = _wc_parts(cameras[c]["w2c"])
        fwds.append(R[2])                                 # world fwd = R_w2c row 2
    pa = []
    for a in range(len(fwds)):
        for b in range(a + 1, len(fwds)):
            d = float(np.clip(np.dot(fwds[a], fwds[b]), -1.0, 1.0))
            pa.append(math.degrees(math.acos(d)))
    pa.sort()
    return round(pa[len(pa) // 2], 1) if pa else 0.0


def load_cameras(sparse_dir):
    sp = Path(sparse_dir)
    intr = pfs.read_cameras_intrinsics(sp / "cameras.bin")
    imgs = pfs.read_images_w2c(sp / "images.bin")
    any_intr = next(iter(intr.values()))
    return [{"name": im["name"], "w2c": np.asarray(im["w2c"], dtype=np.float64),
             "intr": intr.get(im["camera_id"], any_intr)} for im in imgs]


def select_from_sparse(sparse_dir, n, grid=64, near_frac=0.1, far_mult=3.0):
    sp = Path(sparse_dir)
    pts3d = cg.read_points3D_bin(sp / "points3D.bin")
    pts = np.asarray([p["xyz"] for p in pts3d], dtype=np.float64)
    cameras = load_cameras(sp)
    origin, vsize, occupied, diag = voxelize(pts, grid)
    face_c, face_n = exposed_faces(origin, vsize, occupied)
    near, far = vsize * near_frac, diag * far_mult
    W = coverage_matrix(face_c, face_n, cameras, near, far)
    order, cov, gains = greedy_select(W, n)
    F = len(face_c)
    covered = int((cov > 1e-6).sum())
    r = {
        "n_selected": len(order),
        "selected": [cameras[c]["name"] for c in order],
        "n_faces": F,
        "n_occupied_voxels": len(occupied),
        "voxel_size": round(vsize, 5),
        "faces_covered": covered,
        "face_coverage_pct": round(100 * covered / max(1, F), 1),
        "mean_flux": round(float(cov.mean()), 4),
        "median_view_spread_deg": _view_spread(cameras, order),
        "per_step_gain": [round(g, 1) for g in gains],
    }
    return [cameras[c]["name"] for c in order], r


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Voxel-coverage best-N view selector (STO-SCN-103).")
    ap.add_argument("sparse_dir")
    ap.add_argument("--n", type=int, default=24, help="target view count")
    ap.add_argument("--grid", type=int, default=64, help="voxels along the bbox diagonal")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    _, r = select_from_sparse(a.sparse_dir, a.n, a.grid)
    out = Path(a.out) if a.out else Path(a.sparse_dir) / "selection_voxel.json"
    out.write_text(json.dumps(r, indent=2) + "\n")
    print(f"selected {r['n_selected']} views | face-coverage {r['face_coverage_pct']}% "
          f"of {r['n_faces']} faces ({r['n_occupied_voxels']} voxels @ {r['voxel_size']}) | "
          f"mean-flux {r['mean_flux']} | view-spread {r['median_view_spread_deg']} deg")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
