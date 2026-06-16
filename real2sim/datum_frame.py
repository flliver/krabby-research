"""STO-SCN-145 / GOAL-SCN-001 — the camera-derived metric DATUM (gauge-fixing).

Pins the SfM 7-DoF similarity gauge into ONE stable, camera-derived, gravity-aligned, metric
frame so every generated mesh lands in it and operators author boolean cull primitives in meters
against the cameras (the constant). This module is the FRAME; the 1-DoF metric scale comes from
STO-SCN-016 (control distance) and plugs in as `scale`.

Datum recipe (GOAL-SCN-001 / STO-SCN-016 § Decisions D7):
  origin = camera centroid projected onto the gravity ground plane (z = height; optional floor)
  +Z (up) = gravity, from gauge_up.up_from_poses
  +X = spine azimuth (cam[0]->cam[N] projected to the ground plane)
  +Y = Z x X  (right-handed)
  1 unit = 1 m  (scale from STO-SCN-016)

Rigid + isotropic + metric — a 2 m sphere is a 2 m sphere in every scene; NOT a per-axis
normalized unit cube (the cambox is vertically thin -> per-axis normalization distorts shapes).

Pure numpy.
"""
from __future__ import annotations

import numpy as np

import gauge_up


def spine_azimuth(cam_centers, up, order=None):
    """+X = the camera-spine tangent (first->last center) projected onto the ground plane.

    Falls back to the principal horizontal axis (PCA of ground-projected centers) when the spine
    start/end coincide horizontally (e.g. a loop). Returns a unit ground-plane vector.
    """
    C = np.asarray(cam_centers, float)
    if order is not None:
        C = C[np.asarray(order, int)]
    up = np.asarray(up, float); up = up / (np.linalg.norm(up) or 1.0)
    v = C[-1] - C[0]
    e0 = v - up * (v @ up)                       # project tangent onto the ground plane
    n = np.linalg.norm(e0)
    if n < 1e-9:                                 # degenerate -> PCA of ground-projected centers
        Cg = C - np.outer(C @ up, up)
        Cg = Cg - Cg.mean(0)
        _, _, Vt = np.linalg.svd(Cg)
        a = Vt[0] - up * (Vt[0] @ up)
        n = np.linalg.norm(a)
        e0 = a
    return e0 / (n or 1.0)


def build_datum(cam_centers, up, scale=1.0, ground_z=None, order=None):
    """Build the camera-derived metric datum from camera centers + a gravity-up vector.

    Returns a dict: up/e0/e1 axes, R (rows e0,e1,up = solve->datum rotation), origin (solve gauge),
    scale, and `solve_to_datum` (4x4): p_datum = scale * R @ (p_solve - origin).
    `ground_z` (height of the floor along up, e.g. from the orient gauge) drops the origin onto the
    ground plane; default origin = the camera centroid.
    """
    C = np.asarray(cam_centers, float)
    up = np.asarray(up, float); up = up / (np.linalg.norm(up) or 1.0)
    e0 = spine_azimuth(C, up, order)
    e1 = np.cross(up, e0); e1 = e1 / (np.linalg.norm(e1) or 1.0)
    e0 = np.cross(e1, up); e0 = e0 / (np.linalg.norm(e0) or 1.0)   # re-orthonormalize
    R = np.vstack([e0, e1, up])                                   # solve -> datum (rows)

    centroid = C.mean(0)
    if ground_z is None:
        origin = centroid
    else:
        origin = centroid - up * ((centroid @ up) - float(ground_z))   # onto the ground plane

    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = -scale * (R @ origin)
    return {"up": up.tolist(), "e0": e0.tolist(), "e1": e1.tolist(),
            "R": R.tolist(), "origin": origin.tolist(), "scale": float(scale),
            "solve_to_datum": T.tolist(),
            "note": "p_datum = scale * R @ (p_solve - origin); +Z up, +X spine, meters from STO-SCN-016"}


def gauge_fix_from_poses(cam_centers, w2c_list, scale=1.0, ground_z=None, order=None):
    """Convenience: recover gravity-up from the poses (gauge_up) then build the datum."""
    up = gauge_up.up_from_poses(w2c_list)
    return build_datum(cam_centers, up, scale=scale, ground_z=ground_z, order=order)


def to_datum(p_solve, datum):
    """Map a solve-gauge point (or (N,3)) into the datum frame."""
    T = np.asarray(datum["solve_to_datum"], float)
    p = np.asarray(p_solve, float)
    return p @ T[:3, :3].T + T[:3, 3]
