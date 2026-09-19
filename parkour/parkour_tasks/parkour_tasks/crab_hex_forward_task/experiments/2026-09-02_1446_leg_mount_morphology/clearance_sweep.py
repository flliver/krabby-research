#!/usr/bin/env python3
"""Rung (i) leg-leg clearance sweep for leg-mount variants (pure numpy, no Isaac).

Adjacent same-side legs (F-M and M-R) are modelled as capsules -- hip plate (vertical
segment through the plate centre), femur (pivot->knee) and tibia (knee->toe) -- posed by
the Whitworth cam map ``theta_hip = atan2(K sin phi, 1 + K cos phi)`` over cam phase
``phi`` in [0, 2pi), with the neighbour at ``phi + offset`` (offset = pi for the tripod
phasing, 0 for the in-phase worst case). Pitch joints sit at the linkage mid-stroke
defaults (stance) and at a lifted knee (+0.25 rad, the action-scale swing envelope).

Capsule radius is yaw-dependent: the 1-in plywood plank presents its half-thickness
(12.7 mm) to a neighbour when perpendicular to the wall and up to its half-width (63.5 mm)
when yawed: r(theta) = 12.7 mm + 63.5 mm * |sin theta|. Reported: min clearance over the
sweep and the fraction of cam phase with clearance > 20 mm, per variant, pair and offset.

Usage: python clearance_sweep.py [--out clearance_sweep.csv]
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[6]
MDP = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/mdp"
if str(MDP) not in sys.path:
    sys.path.insert(0, str(MDP))
import crab_hex_foot_fk as fk  # noqa: E402

K = fk.dims.YAW_K
HIP_DEFAULT = 0.1105   # crab_hex_linkage.hip_default_rad()
KNEE_DEFAULT = 0.2341  # crab_hex_linkage.knee_default_left_rad()
KNEE_LIFT = 0.25
PLATE_HALF_LEN = fk.dims.HIP_PLATE_LENGTH_IN * fk.dims.IN_TO_M / 2.0
PLATE_W = fk.dims.HIP_PLATE_WIDTH_IN * fk.dims.IN_TO_M
R_THICK = 0.0127
R_WIDE = 0.0635
VARIANTS = [
    ("base", 0.0, 5.5), ("A-20ctrl", -20.0, 5.5), ("B", 0.0, 2.5),
    ("A10", 10.0, 5.5), ("A15", 15.0, 5.5), ("A20", 20.0, 5.5),
    ("A10+B", 10.0, 2.5), ("A15+B", 15.0, 2.5), ("A20+B", 20.0, 2.5),
]
PAIRS = (("FR", "MR"), ("MR", "RR"))


def cam_to_yaw(phi: float) -> float:
    return math.atan2(K * math.sin(phi), 1.0 + K * math.cos(phi))


def mount_yaw_signed(name: str, splay_deg: float) -> float:
    if splay_deg >= 0:
        return fk.mount_yaw(name, splay_deg)
    return -fk.ROW_SIGN[name[0]] * fk.side_sign(name) * math.radians(splay_deg)


def leg_capsules(name, yaw, hip, knee, splay_deg, axis_in):
    """List of (p0, p1, radius) capsules for one leg pose (body frame)."""
    # FK handles the cap; the inward control row bypasses it via a signed mount yaw
    if splay_deg < 0:
        pts = fk.leg_points(name, yaw=yaw + mount_yaw_signed(name, splay_deg), hip=hip, knee=knee,
                            splay_deg=0.0, outer_axis_in=axis_in)
        total_yaw = yaw + mount_yaw_signed(name, splay_deg)
    else:
        pts = fk.leg_points(name, yaw=yaw, hip=hip, knee=knee, splay_deg=splay_deg, outer_axis_in=axis_in)
        total_yaw = yaw + fk.mount_yaw(name, splay_deg)
    r = R_THICK + R_WIDE * abs(math.sin(total_yaw))
    m = pts["mount"]
    sy = fk.side_sign(name)
    c, s = math.cos(total_yaw), math.sin(total_yaw)
    plate_center = m + np.array([-s * sy * PLATE_W / 2.0, c * sy * PLATE_W / 2.0, 0.0])
    plate = (plate_center + np.array([0, 0, -PLATE_HALF_LEN]), plate_center + np.array([0, 0, PLATE_HALF_LEN]), r)
    return [plate, (pts["pivot"], pts["knee"], r), (pts["knee"], pts["toe"], r)]


def seg_seg_distance(p0, p1, q0, q1) -> float:
    """Closest distance between segments p0p1 and q0q1 (Ericson, Real-Time Collision Detection)."""
    d1, d2, r = p1 - p0, q1 - q0, p0 - q0
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    if a <= 1e-12 and e <= 1e-12:
        return float(np.linalg.norm(r))
    if a <= 1e-12:
        s, t = 0.0, float(np.clip(f / e, 0.0, 1.0))
    else:
        c = d1 @ r
        if e <= 1e-12:
            t, s = 0.0, float(np.clip(-c / a, 0.0, 1.0))
        else:
            b = d1 @ d2
            denom = a * e - b * b
            s = float(np.clip((b * f - c * e) / denom, 0.0, 1.0)) if denom > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, float(np.clip(-c / a, 0.0, 1.0))
            elif t > 1.0:
                t, s = 1.0, float(np.clip((b - c) / a, 0.0, 1.0))
    return float(np.linalg.norm((p0 + s * d1) - (q0 + t * d2)))


def min_clearance(caps_a, caps_b) -> float:
    best = float("inf")
    for p0, p1, ra in caps_a:
        for q0, q1, rb in caps_b:
            best = min(best, seg_seg_distance(p0, p1, q0, q1) - ra - rb)
    return best


def sweep(splay_deg, axis_in, pair, offset, knee, n=72):
    vals = []
    for k in range(n):
        phi = 2 * math.pi * k / n
        a = leg_capsules(pair[0], cam_to_yaw(phi), HIP_DEFAULT, knee, splay_deg, axis_in)
        b = leg_capsules(pair[1], cam_to_yaw(phi + offset), HIP_DEFAULT, knee, splay_deg, axis_in)
        vals.append(min_clearance(a, b))
    vals = np.asarray(vals)
    return float(vals.min()), float(np.mean(vals > 0.02))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("clearance_sweep.csv"))
    args = ap.parse_args()
    rows = []
    for tag, splay, axis in VARIANTS:
        for pair in PAIRS:
            for off_name, off in (("tripod_pi", math.pi), ("inphase_0", 0.0)):
                for pose_name, knee in (("stance", KNEE_DEFAULT), ("lifted", KNEE_DEFAULT + KNEE_LIFT)):
                    mn, frac = sweep(splay, axis, pair, off, knee)
                    rows.append({"variant": tag, "splay_deg": splay, "outer_axis_in": axis,
                                 "pair": "-".join(pair), "offset": off_name, "pose": pose_name,
                                 "min_clearance_mm": 1e3 * mn, "frac_phase_gt_20mm": frac})
        worst = min(r["min_clearance_mm"] for r in rows if r["variant"] == tag and r["offset"] == "tripod_pi")
        worst0 = min(r["min_clearance_mm"] for r in rows if r["variant"] == tag and r["offset"] == "inphase_0")
        print(f"{tag:<9} min clearance tripod {worst:7.1f} mm | in-phase {worst0:7.1f} mm", flush=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
