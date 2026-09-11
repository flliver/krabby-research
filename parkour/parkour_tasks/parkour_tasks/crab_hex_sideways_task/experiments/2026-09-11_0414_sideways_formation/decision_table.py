#!/usr/bin/env python3
"""Decision table for the sideways-formation campaign (Part C of the plan; CHARTER.md).

Reads ``scenario_metrics.json`` aggregates (schema: ``crab_hex_forward_task/scripts/gait_eval/
report.py::aggregate``) for the flat-canary / step-onset / obstacle evals of each phase,
recomputes tripod-set duty A/B from ``raw/episode_*.npz`` (contact = ``foot_force_norm >
contact_force_threshold``, masked by ``steady_mask``; tripod A = FL/MR/RL, B = FR/ML/RR in the
harness's ``FOOT_ORDER``), prints the CHARTER's decision table with the forward reference cells
hard-coded from ``crab_hex_forward_task/policy/manifest.yaml``, and evaluates the written 1a gate
(walking-hold tripod >= 0.40 AND completion >= 0.70 on the flat canary; the walking-hold tripod is
the minimum over the holds with a walking command of the per-hold median -- the cmd-0 stand hold
is excluded, see ``walking_tripod`` and the CHARTER's gate section, revised 2026-09-11).

    decision_table.py --campaign-dir <dir> [--phases 1a,2a,2b,2c]
    decision_table.py --eval-dir 1a=<scenario_metrics dir> [--eval-dir 1a:step=<dir> ...]

``--eval-dir`` entries are ``PHASE[:flat|step|obst]=DIR``; the kind defaults to ``flat``. With
``--campaign-dir``, the per-phase eval directories are read from ``state.json`` (``evals.slow/
step/obst.run_dir``, written by ``experiments/tools/run_phases.py``) -- no re-run needed.
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

# policy/manifest.yaml (forward reference of record, A15+B, verified 2026-09-11): completion / falls / episode
# tripod; the 1a flat cell also carries the walking-hold tripod (min over creep 0.633 / low 0.595) read from
# logs/rsl_rl/gait_eval/a15b_lineage/windows_w0_001_A15pB/slow__A15pB/seed001/2026-09-07_03-54-50 -- the
# gate's own units. Later forward windows' per-hold values are not on record here.
FWD = {
    "1a": ("0.86 / 14 / 0.539 (walk 0.595)", "0.69 / 31", "0.68 / 32"),
    "2a": ("0.92 / 8 / 0.513", "0.74 / 26", "0.56 / 44"),
    "2b": ("0.85 / 15 / 0.507", "0.63 / 37", "0.58 / 42"),
    "2c": ("0.81 / 19 / 0.500", "0.68 / 32", "0.66 / 34"),
}
GATE_TRIPOD, GATE_COMPLETION = 0.40, 0.70


def _med(d: dict | None) -> float | None:
    return (d or {}).get("median")


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

    Same rule as ``tools/run_phases.py::walking_tripod`` (kept standalone here): a hold is a walking
    hold when the harness scored it for tracking (``tracking_by_hold[h].ratio.n > 0``, i.e. |cmd| >
    lin_vel_clip 0.2). The episode ``tripod_score`` averages the cmd-0 stand hold in, which scores ~0
    for a head that truly stands -- the forward references' episode numbers include a stand hold
    on which that head kept stepping (0.453), so only the walking holds compare like for like.
    """
    by_hold = {h: _med(v) for h, v in (agg.get("tripod_score_by_hold") or {}).items()}
    track = agg.get("tracking_by_hold") or {}
    # walking holds come from the TRACKING block (one entry per hold of the schedule); the tripod
    # block only carries holds with a scorable gait window, so a walking hold missing there is
    # UNMEASURED -> None (the gate halts), never silently dropped from the minimum
    walking_holds = [h for h, t in track.items() if ((((t or {}).get("ratio") or {}).get("n")) or 0) > 0]
    for h in walking_holds:
        by_hold.setdefault(h, None)
    meds = [by_hold[h] for h in walking_holds]
    if not meds or any(m is None for m in meds):
        return None, by_hold
    return min(meds), by_hold


def cell(agg: dict | None) -> str:
    """``completion / falls/n / episode tripod (walk <min over walking holds>)``. The forward reference
    cells (manifest) carry the episode tripod only (1a also its walking-hold value), so both numbers
    are shown for the sideways rows."""
    if not agg:
        return ""
    tri = _med(agg.get("tripod_score"))
    walk, _ = walking_tripod(agg)
    comp = agg.get("schedule_completion_rate")
    n = agg.get("n_episodes")
    falls = (agg.get("termination_reasons") or {}).get("fall", 0)
    parts = [f"{comp:.2f}" if comp is not None else "n/a", f"{falls}/{n}" if n else str(falls)]
    if tri is not None or walk is not None:
        parts.append((f"{tri:.2f}" if tri is not None else "n/a") + (f" (walk {walk:.2f})" if walk is not None else ""))
    return " / ".join(parts)


def gate_1a(flat_agg: dict) -> tuple[bool, str]:
    """The written 1a gate: walking-hold tripod >= GATE_TRIPOD AND completion >= GATE_COMPLETION."""
    tri, by_hold = walking_tripod(flat_agg)
    comp = flat_agg.get("schedule_completion_rate")
    ok = tri is not None and comp is not None and tri >= GATE_TRIPOD and comp >= GATE_COMPLETION
    tri_s = f"{tri:.3f}" if tri is not None else "n/a"
    comp_s = f"{comp:.3f}" if comp is not None else "n/a"
    holds = ", ".join(f"{h} {v:.3f}" if v is not None else f"{h} n/a" for h, v in by_hold.items()) or "none"
    return ok, (f"walking-hold tripod {tri_s} (min over walking holds; by hold: {holds}; need >= {GATE_TRIPOD}), "
                f"completion {comp_s} (need >= {GATE_COMPLETION})")


def _dirs_from_campaign(campaign_dir: Path) -> dict[str, dict[str, Path]]:
    st = json.loads((campaign_dir / "state.json").read_text())
    dirs: dict[str, dict[str, Path]] = {}
    for recs in st.get("keys", {}).values():
        for phase, rec in recs.items():
            evals = rec.get("evals") or {}
            d = dirs.setdefault(phase, {})
            for kind, key in (("flat", "slow"), ("step", "step"), ("obst", "obst")):
                run_dir = (evals.get(key) or {}).get("run_dir")
                if run_dir:
                    d[kind] = Path(run_dir)
    return dirs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign-dir")
    ap.add_argument("--phases", default="1a,2a,2b,2c")
    ap.add_argument("--eval-dir", action="append", default=[], help="PHASE[:flat|step|obst]=DIR")
    args = ap.parse_args()

    dirs: dict[str, dict[str, Path]] = _dirs_from_campaign(Path(args.campaign_dir)) if args.campaign_dir else {}
    for spec in args.eval_dir:
        ph_kind, path = spec.split("=", 1)
        phase, _, kind = ph_kind.partition(":")
        dirs.setdefault(phase, {})[kind or "flat"] = Path(path)

    print("| phase | flat canary side | fwd | step onset side | fwd | obstacles side | fwd | "
          "tracking | duty A / B | falls (travel frame) | cam | slip |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    gate_lines = []
    for phase in [p.strip() for p in args.phases.split(",") if p.strip()]:
        d = dirs.get(phase, {})
        flat = load_agg(d["flat"]) if "flat" in d else None
        step = load_agg(d["step"]) if "step" in d else None
        obst = load_agg(d["obst"]) if "obst" in d else None
        fwd_flat, fwd_step, fwd_obst = FWD.get(phase, ("", "", ""))
        tracking = _med((flat.get("tracking_by_hold", {}).get("creep") or {}).get("ratio")) if flat else None
        duty_a, duty_b = duty_ab(d["flat"]) if "flat" in d else (None, None)
        duty_str = f"{duty_a:.2f} / {duty_b:.2f}" if duty_a is not None else ""
        falls_str = ", ".join(f"{k}:{v}" for k, v in (flat.get("fall_classes") or {}).items()) if flat else ""
        cam = _med(flat.get("shaft_mean_abs_vel")) if flat else None
        slip = _med(flat.get("slip_ratio")) if flat else None
        tracking_str = f"{tracking:.2f}" if tracking is not None else ""
        cam_str = f"{cam:.2f}" if cam is not None else ""
        slip_str = f"{slip:.2f}" if slip is not None else ""
        print(f"| {phase} | {cell(flat)} | {fwd_flat} | {cell(step)} | {fwd_step} | {cell(obst)} | {fwd_obst} | "
              f"{tracking_str} | {duty_str} | {falls_str} | {cam_str} | {slip_str} |")
        if phase == "1a" and flat is not None:
            ok, why = gate_1a(flat)
            gate_lines.append(f"1a gate: {'PASS' if ok else 'FAIL'} -- {why}")
    for line in gate_lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
