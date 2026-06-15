#!/usr/bin/env python3
"""Recover gravity / world-up from posed cameras — SfM (FastMap) reconstructs only up to an
arbitrary similarity, so the solve gauge has NO absolute orientation. The cameras do carry
it, though: a hand/action camera is held with low ROLL, so every camera's RIGHT axis is
~horizontal ⇒ gravity is the one direction perpendicular to all of them.

This is robust to PITCH (looking up/down) — unlike averaging camera-up, which degrades when
the camera tilts down. Validated on 001-patio: 1.36° from the operator's hand-clicked
building up (avg-camera-up was 3.1°). Assumes low roll; falls back gracefully.

Pure numpy.
"""
from __future__ import annotations

import numpy as np


def up_from_poses(w2c_list) -> list:
    """w2c list (4x4 or 3x4) -> unit world-up (gravity), oriented to match camera-up sense."""
    Rs = [np.asarray(w, dtype=np.float64)[:3, :3] for w in w2c_list]
    rights = np.array([R[0] for R in Rs])        # camera +X (right) in world = row 0 of R_w2c
    ups = np.array([-R[1] for R in Rs])          # camera up = -(camera down) = -row 1
    # gravity ⟂ every right axis ⇒ the least-represented direction among the rights
    _, _, Vt = np.linalg.svd(rights)
    g = Vt[-1] / (np.linalg.norm(Vt[-1]) or 1.0)
    if g.dot(ups.mean(0)) < 0:                    # disambiguate sign toward camera-up
        g = -g
    return g.tolist()


def roll_spread_deg(w2c_list) -> float:
    """Median deviation of camera-right from horizontal (the up-plane) — a confidence gauge.
    Small (a few deg) => the low-roll assumption holds and up_from_poses is trustworthy."""
    up = np.asarray(up_from_poses(w2c_list))
    Rs = [np.asarray(w, dtype=np.float64)[:3, :3] for w in w2c_list]
    rights = np.array([R[0] for R in Rs])
    dev = np.degrees(np.arcsin(np.clip(np.abs(rights @ up), 0, 1)))   # angle of right out of horizontal
    return float(np.median(dev))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    import posed_from_sparse as pfs
    sp = pfs.posed_from_sparse(sys.argv[1])
    w2c = [e["w2c"] for e in sp]
    print("up (gravity):", [round(x, 4) for x in up_from_poses(w2c)])
    print("roll spread (deg, median):", round(roll_spread_deg(w2c), 2))
