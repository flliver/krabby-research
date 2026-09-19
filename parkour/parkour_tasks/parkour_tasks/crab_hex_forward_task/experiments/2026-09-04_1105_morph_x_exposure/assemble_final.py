#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Decision table for the morphology x exposure campaign: one row per configuration with the
carried rung i-iv / rung v columns (morphology campaign CSVs) and the P1 / P2 columns from this
campaign's state.json. Writes final_decision_table.csv and a REPORT block; with --to-results
also appends the block to the morphology campaign's RESULTS.md (cross-link). Isaac venv python."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]  # campaign -> experiments -> crab_hex_forward_task -> parkour_tasks -> parkour_tasks -> parkour -> repo
MORPH = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-09-02_1446_leg_mount_morphology"
ORDER = ["base", "B", "A10", "A15", "A20", "A10+B", "A15+B", "A20+B"]
PREF = {"base": "—", "B": "re-hinge", "A10": "splay only ✓", "A15": "splay only ✓", "A20": "splay only ✓",
        "A10+B": "shims + re-hinge", "A15+B": "shims + re-hinge", "A20+B": "shims + re-hinge"}


def fmt(x, nd=2, suf=""):
    if x is None or (isinstance(x, float) and x != x):
        return "—"
    return f"{x:.{nd}f}{suf}" if isinstance(x, (int, float)) else str(x)


def ratio(x, y, nd=2):
    return "—" if x is None or not y else f"{x / y:.{nd}f}×"


def read_csv(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as fh:
        return {r["config"]: r for r in csv.DictReader(fh)}


def stage_cols(rec: dict | None, ctrl: dict | None, hazard_ref: bool) -> list[str]:
    """Columns for one stage record: slow tripod/compl/track (ratio to golden), slow falls,
    step falls (ratio), obst completion / falls, reach_obst, field_frac, cov[3], fail hazard
    flat / obst, terrain level."""
    if not rec:
        return ["pending"] + ["—"] * 10
    if not rec.get("ckpt"):
        return [rec.get("status", "?")] + ["UNSOUND"] + ["—"] * 9
    ev = rec.get("evals") or {}
    s, p, o = ev.get("slow") or {}, ev.get("step") or {}, ev.get("obst") or {}
    ex = rec.get("exposure") or {}
    cev = (ctrl or {}).get("evals") or {}
    cs, cp = cev.get("slow") or {}, cev.get("step") or {}
    cex = (ctrl or {}).get("exposure") or {}
    return [
        rec.get("status", "?"),
        f"{fmt(s.get('tripod'), 3)} / {fmt(s.get('completion'))} / {fmt(s.get('tracking'), 3)} "
        f"({ratio(s.get('tripod'), cs.get('tripod'))}, {ratio(s.get('completion'), cs.get('completion'))}, {ratio(s.get('tracking'), cs.get('tracking'))})",
        "—" if s.get("falls") is None else f"{s['falls']}/{s.get('n')}",
        "—" if p.get("falls") is None else f"{p['falls']}/{p.get('n')} ({ratio(p['falls'], cp.get('falls'))})",
        "—" if not o else f"{fmt(o.get('completion'))} ({o.get('falls')}/{o.get('n')})",
        fmt(ex.get("reach_obst_frac"), 3), fmt(ex.get("field_frac_mean"), 3), fmt(ex.get("obst_coverage_3"), 3),
        f"{fmt(ex.get('crab_failure_hazard_flat'), 3)} / {fmt(ex.get('crab_failure_hazard_obst'), 3)}"
        + (f" (golden {fmt(cex.get('crab_failure_hazard_flat'), 3)} / {fmt(cex.get('crab_failure_hazard_obst'), 3)})" if cex else ""),
        fmt(ex.get("terrain_levels"), 2),
        fmt(p.get("prefall_tip_p25"), 1, "°"),
    ]


STAGE_HDR = ["status", "slow tripod / compl / track (ratio to golden)", "slow falls", "step falls (ratio)",
             "obst recal2b2w compl (falls)", "reach_obst", "field_frac", "cov[3]", "fail hazard flat / obst (/1k)",
             "terrain level", "step prefall tip p25"]


def main() -> int:
    st = json.loads((HERE / "state.json").read_text()) if (HERE / "state.json").exists() else {}
    stop2 = read_csv(MORPH / "stop2_table.csv")
    stop3 = read_csv(MORPH / "stop3_decision_table.csv")
    p1, p2, s2 = st.get("p1", {}), st.get("p2", {}), st.get("seed2", {})
    hdr = (["config", "hardware", "rung iv 30k falls fwd/step (Δ)", "rung v slow tripod / compl / track (old config)", "rung v step falls (ratio)"]
           + [f"P1 {h}" for h in STAGE_HDR] + [f"P2s1 {h}" for h in STAGE_HDR] + [f"P2s2 {h}" for h in STAGE_HDR]
           + ["seed-2 P2s2 slow / step / obst", "user preference"])
    rows = []
    for cfg in ORDER:
        r2, r3 = stop2.get(cfg, {}), stop3.get(cfg, {})
        row = [cfg, r2.get("hardware", "—"), r2.get("rung iv 30k falls fwd / step (Δ vs base)", "—"),
               f"{r3.get('rung v slow tripod (ratio)', '—')} / {r3.get('rung v slow completion (ratio)', '—')} / {r3.get('rung v slow tracking (ratio)', '—')}",
               r3.get("rung v step falls (ratio)", "—")]
        row += stage_cols(p1.get(cfg), p1.get("base") if cfg != "base" else None, False)
        rec2 = p2.get(cfg) or {}
        base2 = p2.get("base") or {}
        row += stage_cols(rec2.get("stage1"), base2.get("stage1") if cfg != "base" else None, True)
        row += stage_cols(rec2.get("stage2"), base2.get("stage2") if cfg != "base" else None, True)
        sr = (s2.get(cfg) or {}).get("stage2") or {}
        sev = sr.get("evals") or {}
        row.append("—" if not sr else f"{fmt((sev.get('slow') or {}).get('completion'))} / {(sev.get('step') or {}).get('falls', '—')} / {fmt((sev.get('obst') or {}).get('completion'))}")
        row.append(PREF[cfg])
        rows.append(row)
    lines = ["## DECISION TABLE — morphology x training-side fixes (P1: formation + walking slots; P2: + 40-s episodes, 10-s holds; stage 2 on recal2b2w with the promotion equilibrium held)",
             "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    lines += ["- every arm from scratch, seed 3; ratios are variant / the same protocol's golden arm; 0.85× is the canary reference line; rung-v noise floor ≈ 0.06 tripod / 0.05 completion / 5 falls per 100",
              "- rung iv / rung v columns carried from the morphology campaign (stop2_table.csv, stop3_decision_table.csv; rung v = old formation config)",
              "- fail hazards are per 1000 env steps (episode-length neutral); P2 stage-2 terrain level should sit near the lineage's 4–6 or the promotion scaling is re-derived"]
    block = "\n".join(lines)
    print(block)
    with (HERE / "final_decision_table.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(hdr); w.writerows(rows)
    if "--dry-run" not in sys.argv:
        with (HERE / "REPORT.md").open("a") as fh:
            fh.write(block + "\n>>> ENTRY decision table\n\n")
    if "--to-results" in sys.argv:
        with (MORPH / "RESULTS.md").open("a") as fh:
            fh.write("\n" + block + f"\n- source: {HERE}\n>>> ENTRY morph x exposure decision table\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
