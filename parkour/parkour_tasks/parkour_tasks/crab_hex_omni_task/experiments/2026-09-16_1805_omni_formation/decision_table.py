#!/usr/bin/env python3
"""Decision table for the omni-formation campaign (Part D/H of the plan; CHARTER.md).

Reads ``scenario_metrics.json`` aggregates (schema: ``crab_hex_forward_task/scripts/gait_eval/
report.py::aggregate``) for the flat-canary (``slow__omni``) / light-ring-onset (``light__omni``) /
ring-obstacle (``ring__omni``) evals of each phase, recomputes tripod-set duty A/B from
``raw/episode_*.npz`` (contact = ``foot_force_norm > contact_force_threshold``, masked by
``steady_mask``; tripod A = FL/MR/RL, B = FR/ML/RR in the harness's ``FOOT_ORDER``), prints the
CHARTER's decision table with the forward and sideways reference rows hard-coded (their own tasks'
creep/low holds, not phase-matched -- omni is a different command regime), and evaluates the
written gates (``gate_1c`` on 1c; the facing-precision gate from 2a).

    decision_table.py --campaign-dir <dir> [--phases 1a,1b,1c,2a,2b,2c,2c_clockheld]
    decision_table.py --eval-dir 1a=<scenario_metrics dir> [--eval-dir 1a:light=<dir> ...]

``--eval-dir`` entries are ``PHASE[:flat|light|ring]=DIR``; the kind defaults to ``flat``. With
``--campaign-dir``, the per-phase eval directories are read from ``state.json`` (``evals.slow/
light/ring.run_dir``, written by ``experiments/tools/run_phases.py``) -- no re-run needed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FOOT_ORDER = ("FL_Footpad", "FR_Footpad", "ML_Footpad", "MR_Footpad", "RL_Footpad", "RR_Footpad")
TRIPOD_A = ("FL_Footpad", "MR_Footpad", "RL_Footpad")
TRIPOD_B = ("FR_Footpad", "ML_Footpad", "RR_Footpad")
A_IDX = [FOOT_ORDER.index(n) for n in TRIPOD_A]
B_IDX = [FOOT_ORDER.index(n) for n in TRIPOD_B]

TURN_HOLDS = ("turn_l90", "turn_r90")
DIRECTION_HOLDS = ("fwd", "lat_l", "diag", "back")
GATE_1C_DIR_HOLDS = ("lat_l", "diag", "back", "move_turn")   # gate_1c's per-direction floor holds; fwd is
                                                              # already constrained by the tripod/completion criteria

GATE_1C_TRIPOD, GATE_1C_COMPLETION, GATE_1C_YAW_RATE = 0.40, 0.70, 0.05
GATE_1C_DIR_RATIO = 0.3   # per-direction tracking floor (step-16 review decision 3, 2026-09-17)
GATE_FACING_MAX_RAD = 0.30

# Reference rows (Verification 11 / CHARTER caveat: recomputed under the omni definitions from each
# task's OWN manifest and command regime -- not phase-matched to the omni schedule, since neither
# forward nor sideways ever commanded a world direction or a facing target). Printed once, not per
# phase.
FWD_REF = {"label": "fwd ref (forward slow__A15pB creep)", "completion": 0.86, "falls": 14, "n": 100, "walk": 0.633}
SIDE_REF = [
    {"label": "side ref (sideways slow__side seed 2)", "completion": 0.97, "falls": 3, "n": 100, "walk": 0.644},
    {"label": "side ref (sideways slow__side seed 3)", "completion": 0.88, "falls": 12, "n": 100, "walk": 0.548},
]


def _med(d) -> float | None:
    if isinstance(d, dict):
        return d.get("median") if d.get("median") is not None else d.get("mean")
    return d


def load_agg(d: Path) -> dict:
    return json.loads((d / "scenario_metrics.json").read_text())["aggregate"]


def duty_ab(run_dir: Path, threshold: float = 1.0) -> tuple[float | None, float | None]:
    """Mean per-episode tripod-set duty factor, recomputed from the raw contact-force arrays."""
    a_vals, b_vals = [], []
    for f in sorted(run_dir.glob("raw/episode_*.npz")):
        z = np.load(f)
        if "foot_force_norm" not in z.files or "steady_mask" not in z.files:
            continue
        steady = z["steady_mask"]
        if not steady.any():
            continue
        contact = z["foot_force_norm"][steady] > threshold
        duty = contact.mean(axis=0)
        a_vals.append(float(duty[A_IDX].mean()))
        b_vals.append(float(duty[B_IDX].mean()))
    if not a_vals:
        return None, None
    return float(np.mean(a_vals)), float(np.mean(b_vals))


def walking_tripod(agg: dict) -> tuple[float | None, dict[str, float | None]]:
    """Min over the walking holds of the per-hold tripod median, plus the per-hold medians.

    Same rule as ``tools/run_phases.py::walking_tripod`` (kept standalone here, cross-checked by
    ``tests/unit/test_crab_hex_omni_gate.py``): a hold is a walking hold when the harness scored it
    for tracking (``tracking_by_hold[h].ratio.n > 0``, i.e. ``|cmd| > lin_vel_clip`` 0.2). On the
    omni schedule that is ``fwd, lat_l, diag, back, move_turn`` -- ``stand``, ``turn_l90`` and
    ``turn_r90`` command zero speed and are excluded, same as the forward/sideways stand hold.
    """
    by_hold = {h: _med(v) for h, v in (agg.get("tripod_score_by_hold") or {}).items()}
    track = agg.get("tracking_by_hold") or {}
    walking_holds = [h for h, t in track.items() if ((((t or {}).get("ratio") or {}).get("n")) or 0) > 0]
    for h in walking_holds:
        by_hold.setdefault(h, None)
    meds = [by_hold[h] for h in walking_holds]
    if not meds or any(m is None for m in meds):
        return None, by_hold
    return min(meds), by_hold


def facing_by_hold(agg: dict) -> dict[str, dict]:
    """``progress`` is the harness's SIGNED ``yaw_progress_rate`` (mean ``wz * sign(wz_cmd)`` over the
    hold's commanded-turn steps, rad/s toward the command) -- the gated turning quantity; ``net`` the
    signed net rotation toward the hold's target. ``yaw_rate_ratio`` is kept for continuity and is
    NOT gated: a ratio of magnitudes scores gait wobble as turning (2026-09-17, CHARTER Log)."""
    out = {}
    for h, v in (agg.get("facing_by_hold") or {}).items():
        out[h] = {"final": _med(v.get("heading_error_final_abs")), "settle": _med(v.get("settle_time_s")),
                  "progress": _med(v.get("yaw_progress_rate")), "net": _med(v.get("net_yaw_rate_toward")),
                  "yaw_rate_ratio": _med(v.get("yaw_rate_ratio"))}
    return out


def dir_ratio_by_hold(agg: dict, holds: tuple[str, ...] = GATE_1C_DIR_HOLDS) -> dict[str, float | None]:
    tbh = agg.get("tracking_by_hold") or {}
    return {h: _med((tbh.get(h) or {}).get("ratio")) for h in holds}


def gate_1c(agg: dict) -> tuple[bool, str]:
    """The written formation gate (Part H): walking-hold tripod >= 0.40 AND completion >= 0.70 AND
    median yaw-progress rate >= 0.05 rad/s on EACH turn hold, AND (step-16 review decision 3,
    2026-09-17) median tracking ratio (``speed_ratio``) >= 0.3 on EACH of ``GATE_1C_DIR_HOLDS``
    (``lat_l, diag, back, move_turn`` -- ``fwd`` is already constrained above). Twin of
    ``run_phases.py::gate_1c_verdict`` (cross-checked by ``tests/unit/test_crab_hex_omni_gate.py``)."""
    tri, by_hold = walking_tripod(agg)
    comp = agg.get("schedule_completion_rate")
    fac = facing_by_hold(agg)
    rates = {h: (fac.get(h) or {}).get("progress") for h in TURN_HOLDS}
    dirs = dir_ratio_by_hold(agg)
    tri_s = f"{tri:.3f}" if tri is not None else "n/a"
    comp_s = f"{comp:.3f}" if comp is not None else "n/a"
    rate_s = ", ".join(f"{h} {v:.3f}" if v is not None else f"{h} n/a" for h, v in rates.items())
    dir_s = ", ".join(f"{h} {v:.3f}" if v is not None else f"{h} n/a" for h, v in dirs.items())
    failing = [h for h, v in dirs.items() if v is None or v < GATE_1C_DIR_RATIO]
    ok = (tri is not None and comp is not None and all(v is not None for v in rates.values())
          and all(v is not None for v in dirs.values())
          and tri >= GATE_1C_TRIPOD and comp >= GATE_1C_COMPLETION and all(v >= GATE_1C_YAW_RATE for v in rates.values())
          and not failing)
    return ok, (f"walking-hold tripod {tri_s} (need >= {GATE_1C_TRIPOD}), completion {comp_s} "
                f"(need >= {GATE_1C_COMPLETION}), yaw-progress rate {rate_s} (need >= {GATE_1C_YAW_RATE} rad/s), "
                f"per-direction speed_ratio {dir_s} (need >= {GATE_1C_DIR_RATIO} on each of "
                f"{', '.join(GATE_1C_DIR_HOLDS)}" + (f"; failing: {', '.join(failing)})" if failing else ")"))


def gate_facing(agg: dict) -> tuple[bool, str]:
    """Facing-precision gate, enforced from 2a (Part H): median final heading error <= 0.30 rad on
    EACH turn hold."""
    fac = facing_by_hold(agg)
    finals = {h: (fac.get(h) or {}).get("final") for h in TURN_HOLDS}
    final_s = ", ".join(f"{h} {v:.3f}" if v is not None else f"{h} n/a" for h, v in finals.items())
    ok = all(v is not None for v in finals.values()) and max(finals.values()) <= GATE_FACING_MAX_RAD
    return ok, f"final heading error {final_s} (need <= {GATE_FACING_MAX_RAD} rad on each)"


def cell(agg: dict | None) -> str:
    """``completion / falls/n / walking-hold tripod (by hold)``."""
    if not agg:
        return ""
    walk, by_hold = walking_tripod(agg)
    comp = agg.get("schedule_completion_rate")
    n = agg.get("n_episodes")
    falls = (agg.get("termination_reasons") or {}).get("fall", 0)
    parts = [f"{comp:.2f}" if comp is not None else "n/a", f"{falls}/{n}" if n else str(falls)]
    parts.append(f"{walk:.2f}" if walk is not None else "n/a")
    by_hold_s = ", ".join(f"{h} {v:.2f}" if v is not None else f"{h} n/a" for h, v in by_hold.items())
    return " / ".join(parts) + (f" ({by_hold_s})" if by_hold_s else "")


def tracking_cell(agg: dict | None) -> str:
    if not agg:
        return ""
    tbh = agg.get("tracking_by_hold") or {}
    return " / ".join(f"{h} {_med((tbh.get(h) or {}).get('ratio')):.2f}" if _med((tbh.get(h) or {}).get("ratio")) is not None
                       else f"{h} n/a" for h in DIRECTION_HOLDS)


def facing_cell(agg: dict | None) -> str:
    """One table cell for both turn holds, ``final rad / settle s / yaw-progress rad/s`` -- joined
    with ';', NOT '|': this string is embedded inside a markdown table cell, and an unescaped '|'
    would split it into extra columns."""
    if not agg:
        return ""
    fac = facing_by_hold(agg)
    parts = []
    for h in TURN_HOLDS:
        v = fac.get(h) or {}
        parts.append(f"{h} {_fmt(v.get('final'))}/{_fmt(v.get('settle'))}/{_fmt(v.get('progress'))}")
    return "; ".join(parts)


def _fmt(v) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def _dirs_from_campaign(campaign_dir: Path) -> dict[str, dict[str, Path]]:
    st = json.loads((campaign_dir / "state.json").read_text())
    dirs: dict[str, dict[str, Path]] = {}
    for recs in st.get("keys", {}).values():
        for phase, rec in recs.items():
            evals = rec.get("evals") or {}
            d = dirs.setdefault(phase, {})
            for kind, key in (("flat", "slow"), ("light", "light"), ("ring", "ring")):
                run_dir = (evals.get(key) or {}).get("run_dir")
                if run_dir:
                    d[kind] = Path(run_dir)
    return dirs


def _ref_row(ref: dict) -> str:
    walk_s = f"{ref['completion']:.2f} / {ref['falls']}/{ref['n']} / {ref['walk']:.3f}"
    return (f"| {ref['label']} | {walk_s} | | | | | | | | |")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign-dir")
    ap.add_argument("--phases", default="1a,1b,1c,2a,2b,2c,2c_clockheld")
    ap.add_argument("--eval-dir", action="append", default=[], help="PHASE[:flat|light|ring]=DIR")
    args = ap.parse_args()

    dirs: dict[str, dict[str, Path]] = _dirs_from_campaign(Path(args.campaign_dir)) if args.campaign_dir else {}
    for spec in args.eval_dir:
        ph_kind, path = spec.split("=", 1)
        phase, _, kind = ph_kind.partition(":")
        dirs.setdefault(phase, {})[kind or "flat"] = Path(path)

    print("| phase | flat omni (completion / falls/n / walking tripod, by hold) | light ring | ring | "
          "tracking fwd/lat/diag/back | facing final/settle/progress | duty A / B | falls (travel frame) | cam | slip |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    gate_lines = []
    for phase in [p.strip() for p in args.phases.split(",") if p.strip()]:
        d = dirs.get(phase, {})
        flat = load_agg(d["flat"]) if "flat" in d else None
        light = load_agg(d["light"]) if "light" in d else None
        ring = load_agg(d["ring"]) if "ring" in d else None
        duty_a, duty_b = duty_ab(d["flat"]) if "flat" in d else (None, None)
        duty_str = f"{duty_a:.2f} / {duty_b:.2f}" if duty_a is not None else ""
        falls_str = ", ".join(f"{k}:{v}" for k, v in (flat.get("fall_classes") or {}).items()) if flat else ""
        cam = _med(flat.get("shaft_mean_abs_vel")) if flat else None
        slip = _med(flat.get("slip_ratio")) if flat else None
        print(f"| {phase} | {cell(flat)} | {cell(light)} | {cell(ring)} | {tracking_cell(flat)} | "
              f"{facing_cell(flat)} | {duty_str} | {falls_str} | {_fmt(cam)} | {_fmt(slip)} |")
        if phase == "1c" and flat is not None:
            ok, why = gate_1c(flat)
            gate_lines.append(f"1c formation gate: {'PASS' if ok else 'FAIL'} -- {why}")
        if phase in ("2a", "2b") and flat is not None:
            ok, why = gate_facing(flat)
            gate_lines.append(f"{phase} facing gate: {'PASS' if ok else 'FAIL'} -- {why}")
    print(_ref_row(FWD_REF))
    for ref in SIDE_REF:
        print(_ref_row(ref))
    for line in gate_lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
