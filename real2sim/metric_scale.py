"""STO-SCN-144 / STO-SCN-016 — metric-scale recovery core (GOAL-SCN-001).

Authoritative, testable math for recovering the ABSOLUTE METRIC SCALE of a solve gauge from
hand-measured control distances (two-view triangulation), with the DA3 monocular estimate as a
weak prior + gross-error gate only. `match.html` MEASURE mode mirrors this logic in-browser for
live feedback; this module is the source of truth + is unit-tested + is what STO-SCN-016 uses to
apply `s` at the datum/gauge level.

Recipe (STO-SCN-016 § Decisions D1–D7; OLAI corpus
`personal.research/3d-reconstruction/metric-scale-calibration/index.md`):
  - SfM = 7-DoF similarity gauge → scale unobservable from images → external control required.
  - PRIMARY: hand-measured control distance, s = D_measured / d_solve(P1,P2), one scalar on the
    solve Sim3 (datum-level, NOT per-mesh).
  - Each endpoint Pk is triangulated from the operator clicking the SAME feature in ≥2 posed
    photos (closest point of approach between the camera rays).
  - Aggregate ≥2 control distances with a robust MEDIAN IN LOG SPACE; report the spread
    (anisotropy / lens-distortion check).
  - DA3 prior is a GROSS-ERROR GATE only: flag when s_control / median(s_monocular) is outside
    ~1.5x. (The 11.22-vs-3.4 scout split trips this.)

Pure numpy.
"""
from __future__ import annotations

import numpy as np


def pixel_ray(c2w, K, u, v):
    """Back-project an image pixel (u,v) through a posed camera into a world-frame ray.

    c2w: 4x4 (or 3x4) camera-to-world. K: 3x3 intrinsics (fx,fy,cx,cy). Pinhole convention
    matches cull_mesh.py's projection (u = fx*x/z + cx). Returns (origin[3], dir[3] unit).
    """
    c2w = np.asarray(c2w, float)
    K = np.asarray(K, float)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    d_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    R = c2w[:3, :3]
    origin = c2w[:3, 3]
    d_world = R @ d_cam
    d_world = d_world / (np.linalg.norm(d_world) or 1.0)
    return origin.copy(), d_world


def ray_parallelism_deg(d1, d2):
    """Angle between two ray directions (deg). SMALL => near-parallel => weak triangulation."""
    d1 = np.asarray(d1, float); d2 = np.asarray(d2, float)
    d1 = d1 / (np.linalg.norm(d1) or 1.0)
    d2 = d2 / (np.linalg.norm(d2) or 1.0)
    return float(np.degrees(np.arccos(np.clip(abs(d1 @ d2), 0.0, 1.0))))


def triangulate_rays(o1, d1, o2, d2):
    """Closest point of approach between two (skew) rays in a shared frame.

    Returns (point[3], gap, parallax_deg):
      point      = midpoint of the shortest segment connecting the rays (the triangulated 3D point),
      gap        = distance between the rays at closest approach (triangulation confidence; small=good),
      parallax_deg = angle between the rays (small => weak/ill-conditioned triangulation).
    """
    o1 = np.asarray(o1, float); o2 = np.asarray(o2, float)
    d1 = np.asarray(d1, float); d2 = np.asarray(d2, float)
    d1 = d1 / (np.linalg.norm(d1) or 1.0)
    d2 = d2 / (np.linalg.norm(d2) or 1.0)
    r = o1 - o2
    b = float(d1 @ d2)            # a = c = 1 (unit dirs)
    d = float(d1 @ r)
    e = float(d2 @ r)
    denom = 1.0 - b * b          # = a*c - b^2
    if abs(denom) < 1e-9:        # near-parallel: project o1 onto ray 2, take t1=0
        t1, t2 = 0.0, e
    else:
        t1 = (b * e - d) / denom
        t2 = (e - b * d) / denom
    p1 = o1 + t1 * d1
    p2 = o2 + t2 * d2
    point = 0.5 * (p1 + p2)
    gap = float(np.linalg.norm(p1 - p2))
    return point, gap, ray_parallelism_deg(d1, d2)


def scale_from_distance(P1, P2, D_meters):
    """meters-per-solve-unit from one control distance: s = D_measured / |P2-P1|.

    Returns (s, d_solve). d_solve is the gauge-space length between the two triangulated points.
    """
    P1 = np.asarray(P1, float); P2 = np.asarray(P2, float)
    d_solve = float(np.linalg.norm(P2 - P1))
    if d_solve <= 0:
        raise ValueError("degenerate control distance: P1 == P2 in the solve gauge")
    return float(D_meters) / d_solve, d_solve


def aggregate_scales(scales):
    """Robust combine of ≥1 per-distance scales: median IN LOG SPACE + spread (max/min ratio).

    Log space because scale is multiplicative. Spread > ~1.0 flags residual scale anisotropy
    (lens distortion) when the control distances lie on different axes. Returns (s_median, spread).
    """
    s = np.asarray([x for x in scales if x and x > 0], float)
    if len(s) == 0:
        raise ValueError("no positive scales to aggregate")
    s_median = float(np.exp(np.median(np.log(s))))
    spread = float(s.max() / s.min()) if len(s) > 1 else 1.0
    return s_median, spread


def da3_gate(s_control, s_monocular, thresh=1.5):
    """Gross-error gate: is the control-derived scale consistent with the monocular prior?

    s_monocular = list of monocular (DA3) scale ESTIMATES in the SAME parameterization as
    s_control (meters-per-solve-unit). Robust-median them in log space; the gate PASSES iff
    max(s_control/med, med/s_control) <= thresh. Returns (passed, ratio, s_monocular_median);
    passed is None when no prior is available (gate inert — control distance still authoritative).

    NB: the monocular scale_factor → meters-per-solve-unit conversion is a separate (deferred)
    derivation; callers pass already-converted estimates. The gate's job is only to catch a
    gross discrepancy (e.g. the 11.22-vs-3.4 scout outlier) so a mis-clicked pair / wrong length
    gets flagged for human review — it never overrides the control distance.
    """
    sm = np.asarray([x for x in (s_monocular or []) if x and x > 0], float)
    if len(sm) == 0 or not (s_control and s_control > 0):
        return None, None, None
    med = float(np.exp(np.median(np.log(sm))))
    ratio = float(max(s_control / med, med / s_control))
    return bool(ratio <= thresh), ratio, med
