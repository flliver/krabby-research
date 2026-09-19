# SPDX-License-Identifier: BSD-3-Clause
"""Markdown comparison of wave arms vs C0 (from state.json) + the gate ledger of one arm.
Usage: python arm_table.py [ARM_NAME]"""
import json
import sys
from pathlib import Path

st = json.loads((Path(__file__).parent / "state.json").read_text())
arms = dict(st.get("arms", {}))
arms.update({k: v for k, v in st.get("wave2", {}).items() if isinstance(v, dict) and "exposure" in v})
order = [n for n in ["C0", "C3", "C1", "C4", "C2", "C5"] if n in arms] + [n for n in arms if n not in ("C0", "C3", "C1", "C4", "C2", "C5")]
focus = sys.argv[1] if len(sys.argv) > 1 else None


def f(v, d=3):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if v != v else f"{v:.{d}f}"


rows = [
    ("verdict", lambda r: r.get("verdict", "?")),
    ("lever", lambda r: ", ".join(f"{k.replace('KRABBY_', '')}={v}" for k, v in r.get("extra", {}).items()) or "none"),
    ("reach_edge", lambda r: f(r["exposure"].get("reach_edge_frac"))),
    ("reach_obst (gate ≥0.80)", lambda r: f(r["exposure"].get("reach_obst_frac"))),
    ("field_frac (gate ≥0.20)", lambda r: f(r["exposure"].get("field_frac_mean"))),
    ("goals_passed (gate ≥2)", lambda r: f(r["exposure"].get("goals_passed_mean"))),
    ("cov[1]", lambda r: f(r["exposure"].get("obst_coverage_1"))),
    ("cov[2]", lambda r: f(r["exposure"].get("obst_coverage_2"))),
    ("cov[3] (gate ≥0.50)", lambda r: f(r["exposure"].get("obst_coverage_3"))),
    ("cov[4]", lambda r: f(r["exposure"].get("obst_coverage_4"))),
    ("cov[5]", lambda r: f(r["exposure"].get("obst_coverage_5"))),
    ("cov[6] (gate ≥0.20)", lambda r: f(r["exposure"].get("obst_coverage_6"))),
    ("fail flat", lambda r: f(r["exposure"].get("crab_failure_flat"))),
    ("fail flat /1k steps", lambda r: f(r["exposure"].get("crab_failure_hazard_flat"))),
    ("fail obst", lambda r: f(r["exposure"].get("crab_failure_obst"))),
    ("fail obst-RSI", lambda r: f(r["exposure"].get("crab_failure_obst_rsi"))),
    ("fail obst-spread", lambda r: f(r["exposure"].get("crab_failure_obst_spread"))),
    ("ep len (steps)", lambda r: f(r["exposure"].get("eplen"), 0)),
    ("stand time frac (logged)", lambda r: f(r["exposure"].get("stand_frac_actual"))),
    ("stand time frac (corrected)", lambda r: f(float(r["exposure"]["stand_frac_actual"]) * 50.0 * float(r.get("extra", {}).get("KRABBY_EPISODE_S", 20)) / float(r["exposure"]["eplen"]))),
    ("spread frac", lambda r: f(r["exposure"].get("spread_frac_actual"))),
    ("RSI frac", lambda r: f(r["exposure"].get("rsi_frac_actual"))),
    ("terrain level", lambda r: f(r["exposure"].get("terrain_levels"), 2)),
    ("goal_idx (legacy)", lambda r: f(r["exposure"].get("current_goal_idx"))),
    ("mean reward", lambda r: f(r["exposure"].get("mean_reward"), 1)),
    ("value loss", lambda r: f(r["exposure"].get("vloss"), 4)),
    ("canary tripod", lambda r: f(r.get("canary", {}).get("tripod"))),
    ("canary completion", lambda r: f(r.get("canary", {}).get("completion"))),
    ("canary tracking", lambda r: f(r.get("canary", {}).get("tracking"))),
    ("canary creep vx", lambda r: f(r.get("canary", {}).get("creep_vx"))),
    ("canary slip", lambda r: f(r.get("canary", {}).get("slip"))),
    ("obstacle eval completion", lambda r: f(r.get("obst", {}).get("completion"), 2)),
    ("obstacle eval falls/100", lambda r: str(r.get("obst", {}).get("falls", "n/a"))),
]
print("| metric | " + " | ".join(order) + " |")
print("|---|" + "---|" * len(order))
for label, fn in rows:
    cells = []
    for n in order:
        try:
            cells.append(fn(arms[n]))
        except Exception:
            cells.append("n/a")
    print(f"| {label} | " + " | ".join(cells) + " |")
if focus and focus in arms and arms[focus].get("judge"):
    j = arms[focus]["judge"]
    print(f"\nGATES for {focus}: verdict {j.get('verdict')} (kind {j.get('kind')}, exposure status {j.get('exposure_status')})")
    for n in j.get("exposure_notes", []):
        print(f"- exposure: {n}")
    print("- safety: " + ("all gates held" if not j.get("unsafe") else "VIOLATED — " + "; ".join(j["unsafe"])))
    print("- obstacle-tile ceiling: " + (j.get("ceiling") or "under ceiling"))
    if j.get("kind") == "hygiene":
        print(f"- RSI-episode failure share ok: {j.get('rsi_failure_ok')}")
