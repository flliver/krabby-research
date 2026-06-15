#!/usr/bin/env python3
"""STO-SCN-099 — cohesive fusion of per-segment gaussians into one gauge.

The per-segment submaps are globally registered (STO-SCN-098) but still M separate
gaussian clouds. A naive union DOUBLES the geometry in every overlap region — doubled
walls, ghost surfaces at every seam. This stage blends the overlaps into ONE cohesive
gaussian space.

**Seams are a confidence problem, not averaging** (the design note). Each segment observes
its OWN interior well and its boundary poorly; in an overlap, both segments contribute the
same surface. We feather by **camera coverage confidence**: for a gaussian at global point
p, segment k's confidence is a falloff in distance to its nearest camera (the 098 global
poses). A gaussian's opacity is scaled by its segment's confidence NORMALISED across all
segments that cover p:

    w_k(p) = score_k(p) / Σ_j score_j(p),   score_j(p) = exp(-(d_j/r)²)

In a segment's interior only its own score is high → w ≈ 1 (untouched). In an overlap two
segments score equally → w ≈ ½ each → the two overlapping contributions SUM to ~1, not 2:
the doubled wall collapses to a single, cross-faded surface. The higher-confidence segment
wins smoothly at the seam (no hard cut, no smear).

Gaussians are the pipeline's 17-float 3DGS (DC-only SH — so colour is rotation-invariant and
needs no SH rotation under the similarity, the easy case). Transform under a SIM(3) gauge:
position s·R·p+t, log-scale +log s, quaternion composed with R, f_dc/opacity carried.

M=1 (a single tractable space) is a clean pass-through — transform the lone cloud, no feather
— so the single-space path never pays for spine machinery. Output: one .ply in the global
gauge, consumed by STO-SCN-013 (mesh-condition) which owns the subsequent manifolding.

Pure numpy (+ scipy cKDTree for the coverage query). The core operates on arrays so it is
unit-testable without PLYs; `read_ply`/`write_ply` handle the on-disk 3DGS binary.
"""
from __future__ import annotations

import re
import struct
from pathlib import Path

import numpy as np

# 17-float 3DGS vertex layout (this pipeline; DC-only SH).
PLY_PROPS = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
             "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
NPROP = len(PLY_PROPS)
IX = {p: i for i, p in enumerate(PLY_PROPS)}
XYZ = slice(0, 3)
NRM = slice(3, 6)
OPA = IX["opacity"]
SCL = slice(10, 13)
ROT = slice(13, 17)


# ----------------------------------------------------------------- quaternion (wxyz)

def _R_to_quat_wxyz(R) -> np.ndarray:
    R = np.asarray(R, float)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def _quat_mul_wxyz(q, P) -> np.ndarray:
    """Left-multiply each row quaternion in P (N,4 wxyz) by q (4, wxyz)."""
    w0, x0, y0, z0 = q
    w1, x1, y1, z1 = P[:, 0], P[:, 1], P[:, 2], P[:, 3]
    return np.stack([
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1], axis=1)


# ----------------------------------------------------------------- transform / feather

def quat_xyzw_to_R(q) -> np.ndarray:
    """three.js/scout_gauge quaternion [x,y,z,w] -> rotation matrix."""
    x, y, z, w = (float(v) for v in q)
    n = (x * x + y * y + z * z + w * w) ** 0.5 or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def compose_gauge(outer: dict, inner: dict) -> dict:
    """outer ∘ inner as {scale,R,t}: apply `inner` then `outer`. Used to chain the
    STO-SCN-105 scout-gauge (gaussian→segment-solve) with the STO-SCN-098 gauge
    (segment-solve→global) so a DA3-normalized-frame gaussian lands in the global gauge."""
    so, Ro, to = float(outer["scale"]), np.asarray(outer["R"], float), np.asarray(outer["t"], float)
    si, Ri, ti = float(inner["scale"]), np.asarray(inner["R"], float), np.asarray(inner["t"], float)
    return {"scale": so * si, "R": Ro @ Ri, "t": so * (Ro @ ti) + to}


def transform_gaussians(arr: np.ndarray, gauge: dict) -> np.ndarray:
    """Apply a SIM(3) gauge {scale, R(3,3), t(3,)} to a (N,17) 3DGS array -> new array."""
    s = float(gauge["scale"])
    R = np.asarray(gauge["R"], float)
    t = np.asarray(gauge["t"], float)
    out = np.asarray(arr, np.float32).copy()
    out[:, XYZ] = (s * (arr[:, XYZ].astype(float) @ R.T) + t).astype(np.float32)
    n = arr[:, NRM].astype(float)
    if np.any(n):
        out[:, NRM] = (n @ R.T).astype(np.float32)
    out[:, SCL] = (arr[:, SCL].astype(float) + np.log(s)).astype(np.float32)   # log-scale
    qR = _R_to_quat_wxyz(R)
    q = arr[:, ROT].astype(float)
    q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-12, None)
    out[:, ROT] = _quat_mul_wxyz(qR, q).astype(np.float32)
    return out


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _logit(a):
    a = np.clip(a, 1e-6, 1 - 1e-6)
    return np.log(a / (1 - a))


def feather_opacity(arr: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Scale each gaussian's EFFECTIVE alpha by w in [0,1] (3DGS stores logit-opacity).
    a' = w·sigmoid(o); o' = logit(a'). w=1 is a no-op."""
    out = np.asarray(arr, np.float32).copy()
    a = _sigmoid(arr[:, OPA].astype(float))
    out[:, OPA] = _logit(np.clip(w, 0.0, 1.0) * a).astype(np.float32)
    return out


# ----------------------------------------------------------------- confidence weights

def coverage_weights(points: np.ndarray, seg_cameras: dict, owner, radius: float) -> np.ndarray:
    """w_k(p) = score_owner(p) / Σ_j score_j(p), score_j = exp(-(d_j/radius)²), d_j =
    distance to segment j's nearest camera. `points` are seg `owner`'s gaussians (global).
    Regions covered by no camera (all scores ~0) keep w=1 (don't kill isolated geometry —
    halo clipping is STO-SCN-095's job, not fusion's)."""
    from scipy.spatial import cKDTree
    pts = np.asarray(points, float)
    scores = {}
    for j, cams in seg_cameras.items():
        c = np.asarray(cams, float)
        if len(c) == 0:
            scores[j] = np.zeros(len(pts))
            continue
        d, _ = cKDTree(c).query(pts, k=1)
        scores[j] = np.exp(-(d / radius) ** 2)
    denom = np.sum(list(scores.values()), axis=0)
    w = np.where(denom < 1e-9, 1.0, scores[owner] / np.clip(denom, 1e-12, None))
    return w


def _auto_radius(seg_cameras: dict) -> float:
    """Default falloff = median nearest-neighbour camera spacing across all segments
    (the natural overlap scale)."""
    from scipy.spatial import cKDTree
    allc = np.vstack([np.asarray(c, float) for c in seg_cameras.values() if len(c)])
    if len(allc) < 2:
        return 1.0
    d, _ = cKDTree(allc).query(allc, k=2)
    return float(np.median(d[:, 1])) * 3.0 or 1.0


# ----------------------------------------------------------------- fuse

def fuse(segments: dict, *, radius: float | None = None) -> np.ndarray:
    """Confidence-weighted fusion of already-global per-segment gaussians.

    segments: {seg_id: {"gaussians": (N,17) IN GLOBAL GAUGE, "cameras": (C,3) global}}
    Returns one concatenated (ΣN, 17) array with overlap opacities cross-faded so
    overlapping surfaces sum to ~single coverage (no doubled walls). M=1 -> pass-through.
    """
    ids = sorted(segments)
    if len(ids) == 1:
        return np.asarray(segments[ids[0]]["gaussians"], np.float32).copy()
    seg_cams = {k: segments[k]["cameras"] for k in ids}
    r = radius if radius is not None else _auto_radius(seg_cams)
    parts = []
    for k in ids:
        g = np.asarray(segments[k]["gaussians"], np.float32)
        w = coverage_weights(g[:, XYZ], seg_cams, k, r)
        parts.append(feather_opacity(g, w))
    return np.vstack(parts)


# ----------------------------------------------------------------- 3DGS ply io

def read_ply(path) -> np.ndarray:
    """Read a binary-little-endian 3DGS .ply -> (N,17) float32. Validates the 17-prop
    layout (the format the pipeline writes; STO-SCN-095's '17×float32' note)."""
    with open(path, "rb") as f:
        head = b""
        while b"end_header\n" not in head:
            chunk = f.read(1)
            if not chunk:
                raise ValueError(f"{path}: no end_header")
            head += chunk
        n = int(re.search(rb"element vertex (\d+)", head).group(1))
        nprop = head.count(b"property float")
        if nprop != NPROP:
            raise ValueError(f"{path}: expected {NPROP} float props, got {nprop} "
                             f"(non-DC-SH gaussians need an SH-rotation transform — unsupported)")
        buf = np.frombuffer(f.read(n * nprop * 4), dtype="<f4")
    return buf.reshape(n, nprop).copy()


def write_ply(path, arr: np.ndarray) -> None:
    """Write (N,17) float32 as a binary-little-endian 3DGS .ply. Header ends with the
    single '\\n' after end_header, then the raw float32 block (the offset gotcha that
    corrupted a naive rewrite in STO-SCN-095)."""
    arr = np.asarray(arr, np.float32)
    if arr.ndim != 2 or arr.shape[1] != NPROP:
        raise ValueError(f"expected (N,{NPROP}) array, got {arr.shape}")
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(arr)}\n"
              + "".join(f"property float {p}\n" for p in PLY_PROPS)
              + "end_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(np.ascontiguousarray(arr, dtype="<f4").tobytes())


def fuse_files(seg_plys: dict, seg_gauges: dict, seg_cameras: dict, out_ply,
               *, radius: float | None = None) -> dict:
    """End-to-end file fusion: read each segment .ply, transform by its 098 gauge into the
    global frame, confidence-feather the overlaps, concat, write one .ply.

    seg_plys/seg_gauges/seg_cameras keyed by seg_id. Returns a fusion report."""
    segments, n_in = {}, {}
    for k, ply in seg_plys.items():
        g = transform_gaussians(read_ply(ply), seg_gauges[k])
        segments[k] = {"gaussians": g, "cameras": np.asarray(seg_cameras[k], float)}
        n_in[k] = len(g)
    fused = fuse(segments, radius=radius)
    write_ply(out_ply, fused)
    return {"n_segments": len(seg_plys), "n_in": n_in, "n_total_in": sum(n_in.values()),
            "n_fused": int(len(fused)), "radius": radius, "out": str(out_ply)}
