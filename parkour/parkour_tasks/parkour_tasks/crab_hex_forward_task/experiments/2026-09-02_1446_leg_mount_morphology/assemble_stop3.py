#!/usr/bin/env python3
"""STOP 3 assembly: the final decision table (one row per configuration) = rung (v) trained
columns read directly from the formation eval JSONs (control-relative), merged with the
rungs i–iv columns of stop2_table.csv. Writes a markdown block to RESULTS.md (unless
--dry-run) and stop3_decision_table.csv. Run with the isaac venv python."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]  # campaign -> experiments -> crab_hex_forward_task -> parkour_tasks -> parkour_tasks -> parkour -> repo
EV = REPO / "parkour/logs/rsl_rl/gait_eval/leg_mount_morphology/formation"
CONFIGS = ["base", "B", "A10", "A15", "A20", "A10+B", "A15+B", "A20+B"]
PREF = {"base": "—", "B": "re-hinge", "A10": "splay only ✓", "A15": "splay only ✓", "A20": "splay only ✓",
        "A10+B": "shims + re-hinge", "A15+B": "shims + re-hinge", "A20+B": "shims + re-hinge"}


def fmt(x, nd=2, suf=""):
    if x is None or (isinstance(x, float) and x != x):
        return "—"
    return f"{x:.{nd}f}{suf}" if isinstance(x, (int, float)) else str(x)


def latest_agg(arm: str, sid: str) -> dict | None:
    d = EV / arm / sid / "seed001"
    if not d.exists():
        return None
    for r in sorted(d.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        f = r / "scenario_metrics.json"
        if f.exists():
            return json.loads(f.read_text()).get("aggregate")
    return None


def arm_metrics(arm: str) -> dict:
    cfg = "base" if arm == "control" else arm
    out = {}
    for sc in ("slow", "fwd", "step"):
        a = latest_agg(arm.replace("+", "p"), f"{sc}__{cfg.replace('+', 'p')}")
        if not a:
            continue
        fc = a.get("fall_classes", {}); poly = a.get("support_polygon", {})
        nfall = sum(v for k, v in fc.items() if k != "none")
        out[sc] = {
            "tripod": (a.get("tripod_score") or {}).get("median"),
            "completion": a.get("schedule_completion_rate"),
            "tracking": (a.get("tracking_ratio") or {}).get("median"),
            "falls": a["termination_reasons"].get("fall", 0), "n": a["n_episodes"],
            "pitch_fwd_share": fc.get("pitch_fwd", 0) / nfall if nfall else None,
            "pitch_max_abs_p50": ((a.get("orientation") or {}).get("pitch_max_abs") or {}).get("median"),
            "prefall_tip_p25": (poly.get("prefall_tip_angle_fwd_deg") or {}).get("p25"),
            "walk_tip_p50": (poly.get("walking_tip_angle_fwd_deg") or {}).get("median"),
        }
    return out


def ratio(x, y, nd=2):
    return "—" if x is None or not y else f"{x / y:.{nd}f}×"


def main() -> int:
    st = json.loads((HERE / "formation_state.json").read_text()) if (HERE / "formation_state.json").exists() else {"arms": {}}
    stop2 = {}
    if (HERE / "stop2_table.csv").exists():
        with (HERE / "stop2_table.csv").open() as fh:
            for r in csv.DictReader(fh):
                stop2[r["config"]] = r
    ctrl = arm_metrics("control")
    cs, cf, cp = ctrl.get("slow", {}), ctrl.get("fwd", {}), ctrl.get("step", {})
    hdr = ["config", "hardware", "rung i walk tip p50/p10", "rung i lead tip p50", "rung ii standing lead tip Δ",
           "rung iii survivors @0.3/@0.5", "rung iii hip/femur N", "rung iv 30k falls fwd/step (Δ)", "rung iv 20k falls fwd/step (Δ)",
           "rung v status", "smoke fail@2k / coll@2k", "rung v slow tripod (ratio)", "rung v slow completion (ratio)",
           "rung v slow tracking (ratio)", "rung v slow falls Δ", "rung v fwd falls (Δ)", "rung v fwd pitch_max p50 (ratio)",
           "rung v step falls (ratio)", "rung v step pitch-fwd share (ratio)", "rung v step prefall tip p25 Δ", "rung v walk tip p50",
           "user preference"]
    rows = []
    ctrl_row = ["control (anchor, base plant)", "none"] + ["—"] * 7 + ["ok", "—",
        fmt(cs.get("tripod"), 3), fmt(cs.get("completion")), fmt(cs.get("tracking"), 3), f"{cs.get('falls', '—')}",
        f"{cf.get('falls', '—')}", fmt(cf.get("pitch_max_abs_p50")), f"{cp.get('falls', '—')}/{cp.get('n', '—')}",
        fmt(cp.get("pitch_fwd_share")), fmt(cp.get("prefall_tip_p25"), 1, "°"), fmt(cs.get("walk_tip_p50"), 1, "°"), "—"]
    rows.append(ctrl_row)
    for cfg in CONFIGS:
        arm = st["arms"].get(cfg, {}); sm = arm.get("smoke") or {}; s2 = stop2.get(cfg, {})
        m = arm_metrics(cfg) if arm.get("ckpt") else {}
        s, f, p = m.get("slow", {}), m.get("fwd", {}), m.get("step", {})
        rows.append([
            cfg, s2.get("hardware", "—"), s2.get("rung i walk tip p50/p10", "—"), s2.get("rung i lead tip p50", "—"),
            s2.get("rung ii standing lead tip Δ", "—"), s2.get("rung iii survivors/8 @0.3 / @0.5", "—"), s2.get("rung iii hip/femur N", "—"),
            s2.get("rung iv 30k falls fwd / step (Δ vs base)", "—"), s2.get("rung iv 20k falls fwd / step (Δ vs base)", "—"),
            arm.get("status") or "pending", f"{fmt(sm.get('fail_2k'))} / {fmt(sm.get('coll_2k'), 3)}",
            f"{fmt(s.get('tripod'), 3)} ({ratio(s.get('tripod'), cs.get('tripod'))})",
            f"{fmt(s.get('completion'))} ({ratio(s.get('completion'), cs.get('completion'))})",
            f"{fmt(s.get('tracking'), 3)} ({ratio(s.get('tracking'), cs.get('tracking'))})",
            "—" if s.get("falls") is None or cs.get("falls") is None else f"{s['falls']} ({s['falls'] - cs['falls']:+d})",
            "—" if f.get("falls") is None or cf.get("falls") is None else f"{f['falls']} ({f['falls'] - cf['falls']:+d})",
            f"{fmt(f.get('pitch_max_abs_p50'))} ({ratio(f.get('pitch_max_abs_p50'), cf.get('pitch_max_abs_p50'))})",
            "—" if p.get("falls") is None else f"{p['falls']}/{p.get('n')} ({ratio(p['falls'], cp.get('falls'))})",
            f"{fmt(p.get('pitch_fwd_share'))} ({ratio(p.get('pitch_fwd_share'), cp.get('pitch_fwd_share'))})",
            "—" if p.get("prefall_tip_p25") is None or cp.get("prefall_tip_p25") is None else f"{p['prefall_tip_p25']:.1f}° ({p['prefall_tip_p25'] - cp['prefall_tip_p25']:+.1f})",
            fmt(s.get("walk_tip_p50"), 1, "°"), PREF[cfg],
        ])
    lines = ["## STOP 3 — FINAL DECISION TABLE (rungs i–v; the choice is the user's)",
             "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    lines += ["- control = lineage anchor 2026-08-31_03-42-16/model_4999 (seed 3, 5k) on the base plant; variant arms = fresh seed-3 5k formation on their plant with the identical 0–5k config; 'base' arm = same on the golden plant (reproduction sample)",
              "- ratios are variant/control; 0.85× is the plan's reference line on the slow canary; falls are per 100 episodes; 5k is a formation snapshot, not a lineage result",
              "- rung i–iv columns carried over from the STOP 2 table (stop2_table.csv); hardware cost columns there: lateral reach −10/−22/−39 mm per side at 10/15/20°, cos α 0.98/0.97/0.94, sweep overlap unchanged"]
    block = "\n".join(lines)
    print(block)
    with (HERE / "stop3_decision_table.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(hdr); w.writerows(rows)
    if "--dry-run" not in sys.argv:
        with (HERE / "RESULTS.md").open("a") as fh:
            fh.write("\n" + block + "\n>>> ENTRY stop 3 decision table\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
