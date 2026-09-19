#!/usr/bin/env python3
"""Rung (i) kinematic screen: support-polygon margins of recorded gaits under leg-mount variants.

For every stored eval episode (foot positions, root pose, foot forces, failure flags), each
foot is moved to where the SAME joint angles would place it on a variant plant -- a rigid
rotation about the leg's original yaw axis by the mount splay plus a translation of the
axis along x (exact: the whole leg chain rotates with its mount; see
crab_hex_foot_fk.transform_recorded_foot). The transform is applied in the TRUE body frame
(full root quaternion), so tilted pre-fall frames are handled correctly. The stability
metrics are then recomputed per variant over walking frames and over the 1.0 s pre-fall
windows.

Premise (stated in every table): same joint angles as the recorded policy -- a lower bound on
what a policy re-trained on the variant could exploit, not a prediction of trained behaviour.

Usage: python kinematic_screen.py [--eval-dir DIR ...] [--out screen.csv]
Pure numpy; no Isaac.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[6]
MDP = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/mdp"
SCRIPTS = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/scripts"
for p in (MDP, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import crab_hex_foot_fk as fk  # noqa: E402
from gait_eval import metrics as M  # noqa: E402

EVAL_ROOT = REPO / "parkour/logs/rsl_rl/gait_eval"
DEFAULT_EVAL_DIRS = {
    "30k_flat": EVAL_ROOT / "gait_income_phaseout/flat_walk_slow_v2/seed001/2026-09-02_06-52-54",
    "30k_light_obst": EVAL_ROOT / "gait_income_phaseout_obst_light/flat_walk_slow_v2/seed001/2026-09-02_16-40-21",
    "20k_ref_obst": EVAL_ROOT / "gated_lineage_obst/flat_walk_slow_v2/seed001/2026-08-31_17-13-40",
}
# eval harness foot order (metrics.FOOT_ORDER mirrors the contact-sensor body order that
# eval_crab_hex_gait.py asserts against CRAB_HEX_FOOTPAD_BODY_NAMES)
FOOT_ORDER = tuple(n.split("_")[0] for n in M.FOOT_ORDER)  # "FL_Footpad" -> "FL"
assert set(FOOT_ORDER) == set(fk.LEG_NAMES), FOOT_ORDER
VARIANTS = [
    ("base", 0.0, 5.5), ("A-20ctrl", -20.0, 5.5), ("B", 0.0, 2.5),
    ("A10", 10.0, 5.5), ("A15", 15.0, 5.5), ("A20", 20.0, 5.5),
    ("A10+B", 10.0, 2.5), ("A15+B", 15.0, 2.5), ("A20+B", 20.0, 2.5),
]
WALK_CMD_MIN = 0.2


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """(T, 4) wxyz -> (T, 3, 3) rotation matrices (body -> world)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - z * w); R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w); R[:, 2, 1] = 2 * (y * z + x * w); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def mount_yaw_signed(name: str, splay_deg: float) -> float:
    """Allow the negative (inward / insect-layout) control row: bypass the cap for splay < 0."""
    if splay_deg >= 0:
        return fk.mount_yaw(name, splay_deg)
    return -fk.ROW_SIGN[name[0]] * fk.side_sign(name) * math.radians(splay_deg)


def transform_feet(foot_pos_w, root_pos_w, root_quat_w, splay_deg, outer_axis_in):
    """Apply the rigid mount transform per foot in the body frame; return world positions."""
    R = quat_to_rot(root_quat_w)                      # body -> world
    rel_b = np.einsum("tji,tkj->tki", R, foot_pos_w - root_pos_w[:, None, :])  # R^T (f - r)
    out_b = np.empty_like(rel_b)
    for i, name in enumerate(FOOT_ORDER):
        p0 = fk.mount_point(name, fk.LEGACY_OUTER_AXIS_IN)  # recordings are golden-plant (2026-09-09 note)
        dx = fk.mount_x(name, outer_axis_in) - fk.mount_x(name, fk.LEGACY_OUTER_AXIS_IN)
        a = mount_yaw_signed(name, splay_deg)
        c, s = math.cos(a), math.sin(a)
        rel = rel_b[:, i, :] - p0
        rot = np.stack([c * rel[:, 0] - s * rel[:, 1], s * rel[:, 0] + c * rel[:, 1], rel[:, 2]], axis=-1)
        out_b[:, i, :] = rot + p0 + np.array([dx, 0.0, 0.0])
    return root_pos_w[:, None, :] + np.einsum("tij,tkj->tki", R, out_b)


def geometry_columns(splay_deg: float, outer_axis_in: float) -> dict:
    """Pure-geometry columns: fore-aft half-base, sweep overlap, reach loss, cos alpha, sideways."""
    a = math.radians(abs(splay_deg))
    radius = fk.PIVOT_OUTBOARD_M + fk.FEMUR_LEN_M
    half_base = fk.mount_x("RR", outer_axis_in) + (radius * math.sin(a) if splay_deg > 0 else 0.0)
    spacing = fk.mount_x("RR", outer_axis_in)  # M at 0 -> F/R spacing
    sweep = fk.toe_x_sweep_m()
    # neutral toe x of the outer leg moves by radius*sin(alpha) away from the mid leg (outward)
    gap = spacing + (radius * math.sin(a) if splay_deg > 0 else -radius * math.sin(a))
    overlap = 2 * sweep - gap
    return {
        "half_base_m": half_base,
        "sweep_overlap_m": overlap,
        "lateral_reach_loss_mm": 1e3 * radius * (1.0 - math.cos(a)),
        "cos_alpha": math.cos(a),
        "sideways_reachable": fk.perpendicular_reachable(abs(splay_deg)),
    }


def load_episode(path: Path) -> dict:
    d = np.load(path)
    return {k: d[k] for k in d.files}


def screen_dir(eval_dir: Path, variants) -> list[dict]:
    files = sorted(glob.glob(str(eval_dir / "**/raw/episode_*.npz"), recursive=True))
    if not files:
        raise SystemExit(f"no raw episodes under {eval_dir}")
    rows = []
    for tag, splay, axis in variants:
        walk_tip, walk_lead, walk_fwd, walk_neg = [], [], [], []
        pre_tip, pre_lead, pre_fwd, pre_neg, pre_under = [], [], [], [], []
        n_fail = 0
        for f in files:
            e = load_episode(Path(f))
            dt = float(e["dt"])
            fp = transform_feet(e["foot_pos_w"], e["root_pos_w"], e["root_quat_w"], splay, axis)
            walking = e["steady_mask"].astype(bool) & (np.abs(e["cmd_applied"][:, 0]) >= WALK_CMD_MIN)
            res = M.support_polygon_metrics(
                fp, e["foot_force_norm"], e["root_pos_w"], e["root_quat_w"], dt=dt,
                crab_failure=e["crab_failure"], walking_mask=walking, return_series=True,
            )
            s = res["series"]
            m = walking & np.isfinite(s["fwd_ray_margin_m"])
            walk_tip.extend(s["tip_angle_fwd_deg"][m]); walk_lead.extend(s["lead_contact_tip_deg"][m])
            walk_fwd.extend(s["fwd_ray_margin_m"][m]); walk_neg.extend(s["fwd_ray_margin_m"][m] < 0)
            if e["crab_failure"].any():
                n_fail += 1
                pm = s["prefall_mask"]
                pre_under.extend(s["n_loaded"][pm] < 3)
                pm2 = pm & np.isfinite(s["fwd_ray_margin_m"])
                pre_tip.extend(s["tip_angle_fwd_deg"][pm2]); pre_lead.extend(s["lead_contact_tip_deg"][pm2])
                pre_fwd.extend(s["fwd_ray_margin_m"][pm2]); pre_neg.extend(s["fwd_ray_margin_m"][pm2] < 0)

        def pct(v, q):
            return float(np.percentile(v, q)) if len(v) else float("nan")

        row = {"eval": eval_dir.parent.parent.parent.name + "/" + eval_dir.name, "variant": tag,
               "splay_deg": splay, "outer_axis_in": axis, "n_episodes": len(files), "n_falls": n_fail,
               "walk_tip_fwd_p50": pct(walk_tip, 50), "walk_tip_fwd_p10": pct(walk_tip, 10),
               "walk_lead_tip_p50": pct(walk_lead, 50), "walk_lead_tip_p10": pct(walk_lead, 10),
               "walk_fwd_margin_p50": pct(walk_fwd, 50), "walk_frac_neg": float(np.mean(walk_neg)) if walk_neg else float("nan"),
               "prefall_tip_fwd_p50": pct(pre_tip, 50), "prefall_tip_fwd_p10": pct(pre_tip, 10),
               "prefall_lead_tip_p50": pct(pre_lead, 50), "prefall_lead_tip_p10": pct(pre_lead, 10),
               "prefall_fwd_margin_p50": pct(pre_fwd, 50), "prefall_frac_neg": float(np.mean(pre_neg)) if pre_neg else float("nan"),
               "prefall_frac_underdetermined": float(np.mean(pre_under)) if pre_under else float("nan")}
        row.update(geometry_columns(splay, axis))
        rows.append(row)
        print(f"{row['eval']:<48} {tag:<9} walk tip p50/p10 {row['walk_tip_fwd_p50']:6.1f}/{row['walk_tip_fwd_p10']:6.1f} "
              f"lead {row['walk_lead_tip_p50']:6.1f}/{row['walk_lead_tip_p10']:6.1f} | prefall tip {row['prefall_tip_fwd_p50']:6.1f}/"
              f"{row['prefall_tip_fwd_p10']:6.1f} lead {row['prefall_lead_tip_p50']:6.1f}/{row['prefall_lead_tip_p10']:6.1f} "
              f"neg {row['prefall_frac_neg']:.2f}", flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-dir", action="append", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("kinematic_screen.csv"))
    args = ap.parse_args()
    dirs = args.eval_dir or list(DEFAULT_EVAL_DIRS.values())
    rows = []
    for d in dirs:
        rows.extend(screen_dir(Path(d), VARIANTS))
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
