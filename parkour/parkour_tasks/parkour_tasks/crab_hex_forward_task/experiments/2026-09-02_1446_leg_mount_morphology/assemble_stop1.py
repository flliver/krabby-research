#!/usr/bin/env python3
"""STOP 1 assembly for rung (ii): offline FK-vs-Isaac cross-check of every variant's settled
pose, standing support-polygon tip angles, the d^2 stiffness proxy from settled foot
positions, spawn-Z check, and the sign-off video list. Reads statics/statics_<tag>.{json,npz}.
Run with the isaac venv python (numpy + repo modules)."""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]  # campaign -> experiments -> crab_hex_forward_task -> parkour_tasks -> parkour_tasks -> parkour -> repo
TASK = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task"
sys.path.insert(0, str(TASK / "scripts"))
sys.path.insert(0, str(TASK / "mdp"))
from gait_eval import metrics as M  # noqa: E402
import crab_hex_foot_fk as FK  # noqa: E402

TAGS = ["base", "B", "A10", "A15", "A20", "A10_B", "A15_B", "A20_B"]
LEGS = ("FL", "FR", "ML", "MR", "RL", "RR")


def variant_params(usd: str) -> tuple[float, float]:
    m = re.search(r"splay(\d+)_axis(\d)p(\d)in", usd or "")
    if not m:
        return FK.LEGACY_SPLAY_DEG, FK.LEGACY_OUTER_AXIS_IN  # base = legacy golden (2026-09-09: FK defaults moved to A15+B)
    return float(m.group(1)), float(f"{m.group(2)}.{m.group(3)}")


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def fk_residuals(js: dict, z: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    splay, axis_in = variant_params(js.get("usd", ""))
    jn, bn = js["joint_names"], js["body_names"]
    q = np.asarray(z["joint_pos"], dtype=np.float64)
    R = quat_to_rot(np.asarray(z["root_quat_w"], dtype=np.float64))
    root = np.asarray(z["root_pos_w"], dtype=np.float64)
    out = {}
    for leg in LEGS:
        yaw = q[jn.index(f"{leg}_Body_Hip_RevoluteJoint")]
        hip = q[jn.index(f"{leg}_Hip_Femur_RevoluteJoint")]
        knee = q[jn.index(f"{leg}_Femur_Tibia_RevoluteJoint")]
        if leg not in FK.LEFT:  # right legs carry a 180-deg pitch-frame flip (module docstring)
            knee = -knee
        toe_fk = FK.leg_points(leg, yaw, hip, knee, splay_deg=splay, outer_axis_in=axis_in)["toe"]
        foot_w = np.asarray(z["body_pos_w"][bn.index(f"{leg}_Footpad")], dtype=np.float64)
        foot_b = R.T @ (foot_w - root)
        out[leg] = foot_b - toe_fk
    return out


def foot_pos(js: dict, z) -> np.ndarray:
    """Last-settle footpad positions (6, 3) from the full body_pos_w buffer, FL..RR order."""
    return np.asarray([z["body_pos_w"][js["body_names"].index(f"{leg}_Footpad")] for leg in LEGS], dtype=np.float64)


def standing_polygon(js: dict, z, dt: float = 0.02) -> dict:
    fp = foot_pos(js, z)[None]
    ff = np.asarray(z["foot_force_N"])[None]
    return M.support_polygon_metrics(
        fp, ff, np.asarray(z["root_pos_w"])[None], np.asarray(z["root_quat_w"])[None], dt=dt,
        crab_failure=np.zeros(1, dtype=bool), walking_mask=np.ones(1, dtype=bool),
        com_pos_w=np.asarray(z["com_w"])[None],
    )["walking"]


def stiffness_proxy(js: dict, z) -> float:
    """d^2 heuristic on the settled geometry: sum over loaded feet of (x_foot - x_com)^2 in the
    heading frame (leg-PD pitch lever arms). Relative use only."""
    R = quat_to_rot(np.asarray(z["root_quat_w"], dtype=np.float64))
    yaw = math.atan2(R[1, 0], R[0, 0])
    c, s = math.cos(yaw), math.sin(yaw)
    fp = foot_pos(js, z)
    com = np.asarray(z["com_w"], dtype=np.float64)
    loaded = np.asarray(z["foot_force_N"]) >= 50.0
    x = (fp[:, 0] - com[0]) * c + (fp[:, 1] - com[1]) * s
    return float(np.sum(x[loaded] ** 2))


def main() -> int:
    rows, base = [], None
    for tag in TAGS:
        jp, npz = HERE / "statics" / f"statics_{tag}.json", HERE / "statics" / f"statics_{tag}.npz"
        if not jp.exists() or not npz.exists():
            rows.append({"tag": tag, "missing": True})
            continue
        js, z = json.loads(jp.read_text()), np.load(npz, allow_pickle=True)
        res = fk_residuals(js, z)
        poly = standing_polygon(js, z)
        row = {
            "tag": tag, "missing": False,
            "fk_res_mm": {leg: float(np.linalg.norm(v) * 1000) for leg, v in res.items()},
            "fk_res_vec": {leg: v for leg, v in res.items()},
            "tip_fwd_deg": poly["tip_angle_fwd_deg"]["p50"],
            "lead_tip_deg": poly["lead_contact_tip_deg"]["p50"],
            "fwd_margin_m": poly["fwd_ray_margin_m"]["p50"],
            "k_proxy": stiffness_proxy(js, z),
            "root_z": js["root_z_mean"], "n_upright": js["n_upright"], "n_pen": js["n_penetrating"],
            "leg_contact": max(js["leg_link_contact_max_N"], js["sweep"]["leg_link_contact_max_N"]),
            "A_share_sd": js["A_share_sd"], "tilt_max": max(t["tilt_max_during_settle"] for t in js["trials"]),
            "sweep_min_foot_z": js["sweep"]["min_foot_z"],
            "feet_loaded": int(np.sum(np.asarray(z["foot_force_N"]) >= 50.0)),
            "video": sorted((HERE / "statics" / f"video_{tag}").glob("*.mp4")),
        }
        if tag == "base":
            base = row
        rows.append(row)
    lines = ["## RUNG (ii) — cross-check + scored columns (offline, from statics npz)",
             "| config | sound? | FK residual max mm (Δ vs base) | standing polygon tip (feet loaded) | standing lead-contact tip (Δ vs base) | fwd margin m | K proxy ratio (d² model) | root_z Δ mm | tilt max settle deg | A-share sd | video |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["missing"]:
            lines.append(f"| {r['tag']} | MISSING | | | | | | | | | |")
            continue
        sound = r["n_upright"] >= 19 and r["n_pen"] == 0 and r["leg_contact"] < 15.0
        fkmax = max(r["fk_res_mm"].values())
        d_fk = fkmax - max(base["fk_res_mm"].values()) if base else float("nan")
        d_lead = r["lead_tip_deg"] - base["lead_tip_deg"] if base and r["lead_tip_deg"] is not None else float("nan")
        kr = r["k_proxy"] / base["k_proxy"] if base and base["k_proxy"] else float("nan")
        dz = (r["root_z"] - base["root_z"]) * 1000 if base else float("nan")
        gate = "strong" if d_lead >= 8.0 else ("pass" if d_lead >= 4.0 else ("—" if r["tag"] == "base" else "miss"))
        lines.append(
            f"| {r['tag']} | {'yes' if sound else 'NO'} | {fkmax:.1f} ({d_fk:+.1f}) | {r['tip_fwd_deg']:.1f}° ({r['feet_loaded']}) | "
            f"{r['lead_tip_deg']:.1f}° ({d_lead:+.1f}°, {gate}) | {r['fwd_margin_m']:.3f} | {kr:.2f}× | {dz:+.1f} | "
            f"{math.degrees(r['tilt_max']):.1f} | {100 * r['A_share_sd']:.1f}% | {r['video'][0].name if r['video'] else 'none'} |")
    lines += [
        "- FK residual = |Isaac footpad (body frame) − offline FK toe| at the last settle; the base row's value is the footpad-vs-toe-point offset common to all rows, so Δ vs base is the mount-transform check (plan tolerance 5 mm)",
        "- standing lead-contact tip gate: pass ≥ base +4°, strong ≥ base +8° (plan: baseline +8°); K proxy = Σ(x_foot − x_com)² over loaded feet, a geometric d² stand-in, relative only",
        "- root_z Δ within ±10 mm keeps the default KRABBY_HEX_SPAWN_Z for the variant",
        "- standing polygon tip is a single-frame value from the last settle (a 5-foot settle exposes a diagonal forward edge); the lead-contact tip is the gate statistic",
    ]
    block = "\n".join(lines)
    print(block)
    if "--dry-run" not in sys.argv:
        with (HERE / "RESULTS.md").open("a") as fh:
            fh.write("\n" + block + "\n>>> ENTRY rung ii crosscheck\n")
    (HERE / "stop1_rows.json").write_text(json.dumps(
        [{k: (v if k not in ("video", "fk_res_vec") else ([str(p) for p in v] if k == "video" else {l: [float(x) for x in a] for l, a in v.items()}))
          for k, v in r.items()} for r in rows], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
