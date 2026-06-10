"""gauge_align — canonical similarity (gauge) alignment between camera sets.

STO-SCN-048 (Photo Spine Pipeline). One implementation of the
Umeyama/Procrustes-with-scale solve used everywhere camera sets from
different SfM solves must share a frame:

  - photo-spine chunk stitching (batched_sfm.py — primary consumer)
  - comparison-view injection (build_blender_scene.py — inline copy,
    consolidation tracked in STO-SCN-048; this module is the canonical)
  - viewer virtual-camera alignment (camera_viewer/viewer.py — same)

Pure numpy. No Blender, no torch.
"""
from __future__ import annotations

import numpy as np


def umeyama(P: np.ndarray, Q: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Solve s, R, t such that  s * R @ P_i + t ≈ Q_i  (least squares).

    P, Q: (N, 3) corresponding points (N ≥ 3).
    Returns (scale, R(3,3), t(3,)). det(R) = +1 (reflections corrected).
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    if P.shape != Q.shape or P.shape[0] < 3 or P.shape[1] != 3:
        raise ValueError(f"need matching (N>=3, 3) point sets, got {P.shape} vs {Q.shape}")
    cP, cQ = P.mean(axis=0), Q.mean(axis=0)
    Pc, Qc = P - cP, Q - cQ
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    var_P = float((Pc * Pc).sum())
    scale = float((np.diag(D) * S).sum() / var_P) if var_P > 0 else 1.0
    t = cQ - scale * R @ cP
    return scale, R, t


def residuals(P: np.ndarray, Q: np.ndarray, scale: float, R: np.ndarray,
              t: np.ndarray) -> np.ndarray:
    """Per-point |s·R·P + t − Q| (meters in Q's gauge)."""
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    return np.linalg.norm((scale * (P @ R.T) + t) - Q, axis=1)


def apply_to_cams2world(cams2world: np.ndarray, scale: float, R: np.ndarray,
                        t: np.ndarray) -> np.ndarray:
    """Map (N,4,4) cam→world transforms from gauge P into gauge Q.

    Rotation columns get R (no scale — camera orientation is unitless);
    translation columns get the full similarity s·R·x + t.
    """
    M = np.asarray(cams2world, dtype=np.float64).copy()
    M[:, :3, :3] = np.einsum("ij,njk->nik", R, M[:, :3, :3])
    M[:, :3, 3] = scale * np.einsum("ij,nj->ni", R, M[:, :3, 3]) + t
    return M


def align_camera_sets(src_positions: np.ndarray, dst_positions: np.ndarray,
                      max_residual: float | None = None,
                      src_rotations: np.ndarray | None = None,
                      dst_rotations: np.ndarray | None = None,
                      ) -> dict:
    """Align src gauge onto dst gauge through corresponding cameras.

    POSITION-ONLY Umeyama is rotation-ambiguous when the shared camera
    centers are near-coplanar or near-collinear — which is the COMMON
    case for spine overlaps (orbit arcs, walking paths). Caught by the
    2026-06-10 synthetic test: dtu's orbit ring recovered positions to
    2e-15 m but orientations were off 2.55°. When rotations (N,3,3
    cam→world) are provided, the solve is augmented with synthetic
    points along each camera's optical axes (two-pass: positions-only
    first to estimate scale, then augmented), which pins the rotation
    even for degenerate center geometry.

    Returns {scale, R, t, residuals, max_residual, mean_residual}
    (residuals over the CENTERS only — the gate semantics stay in
    meters of camera-position disagreement).
    Raises RuntimeError when max_residual is exceeded (per-stitch HARD
    GATE: a bad stitch must fail loudly, never propagate).
    """
    P = np.asarray(src_positions, dtype=np.float64)
    Q = np.asarray(dst_positions, dtype=np.float64)
    s, R, t = umeyama(P, Q)

    if src_rotations is not None and dst_rotations is not None:
        Rs = np.asarray(src_rotations, dtype=np.float64)
        Rd = np.asarray(dst_rotations, dtype=np.float64)
        # Offset length ~ the overlap's spatial extent (well-conditioned).
        d_dst = max(float(np.linalg.norm(Q - Q.mean(axis=0), axis=1).mean()), 1e-6)
        d_src = d_dst / s if s > 0 else d_dst
        aug_P = [P]
        aug_Q = [Q]
        for axis in (2, 1):  # optical (z) and up (y) axes
            aug_P.append(P + Rs[:, :, axis] * d_src)
            aug_Q.append(Q + Rd[:, :, axis] * d_dst)
        s, R, t = umeyama(np.vstack(aug_P), np.vstack(aug_Q))

    res = residuals(P, Q, s, R, t)
    out = {"scale": s, "R": R, "t": t, "residuals": res,
           "max_residual": float(res.max()), "mean_residual": float(res.mean())}
    if max_residual is not None and out["max_residual"] > max_residual:
        raise RuntimeError(
            f"stitch residual gate: max {out['max_residual']:.4f} m exceeds "
            f"allowed {max_residual:.4f} m (mean {out['mean_residual']:.4f}). "
            f"Overlap poses disagree — refuse to chain a bad gauge.")
    return out


def consensus_align(src_positions: np.ndarray, dst_positions: np.ndarray,
                    rel_tol: float = 0.02,
                    min_consensus: int = 6,
                    min_consensus_frac: float = 0.25,
                    src_rotations: np.ndarray | None = None,
                    dst_rotations: np.ndarray | None = None,
                    ) -> dict:
    """Robust gauge alignment: iteratively trim disagreeing correspondences.

    Real chunk solves contain badly-registered frames (blurry /
    featureless inputs) whose poses are outright wrong; a plain
    least-squares solve over the full overlap is poisoned by them
    (005-meadow 2026-06-10: 01↔02 full-overlap mean residual 3.7 in a
    span-39 gauge). This trims the worst correspondence one at a time
    until every survivor's residual is below the RELATIVE gate

        gate = rel_tol × mean spread of the surviving dst points

    (absolute gates are meaningless across arbitrary per-chunk gauge
    scales — the 0.10 "m" default of the first stitch attempt was a
    unit error).

    Hard failures (RuntimeError):
      - consensus shrinks below min_consensus frames, or below
        min_consensus_frac of the overlap, without converging —
        the link is BROKEN, not noisy; refuse to chain it.

    Returns the align_camera_sets dict plus:
      consensus_idx (indices into the input arrays), n_overlap,
      consensus_frac, gate, outlier_idx.
    """
    P = np.asarray(src_positions, dtype=np.float64)
    Q = np.asarray(dst_positions, dtype=np.float64)
    n = len(P)
    Rs = None if src_rotations is None else np.asarray(src_rotations, np.float64)
    Rd = None if dst_rotations is None else np.asarray(dst_rotations, np.float64)
    keep = np.arange(n)
    floor = max(min_consensus, int(np.ceil(min_consensus_frac * n)))
    while True:
        a = align_camera_sets(
            P[keep], Q[keep],
            src_rotations=None if Rs is None else Rs[keep],
            dst_rotations=None if Rd is None else Rd[keep])
        r = a["residuals"]
        spread = float(np.linalg.norm(Q[keep] - Q[keep].mean(0), axis=1).mean())
        gate = rel_tol * spread
        if r.max() <= gate:
            break
        if len(keep) - 1 < floor:
            raise RuntimeError(
                f"consensus collapse: {len(keep)}/{n} overlap frames still "
                f"disagree (max {r.max():.4f} > gate {gate:.4f} = "
                f"{rel_tol:.3f}×spread {spread:.4f}); floor {floor}. "
                f"Link is broken — refuse to chain.")
        keep = keep[np.argsort(r)[:-1]]
    a["consensus_idx"] = keep
    a["n_overlap"] = n
    a["consensus_frac"] = len(keep) / n
    a["gate"] = gate
    a["outlier_idx"] = np.setdiff1d(np.arange(n), keep)
    return a
