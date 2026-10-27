#!/usr/bin/env python3
"""Step 4 (CPU): the anchors -> every anchor-relative gate clause as a NUMBER, + the step-4 STOP checks.

Anchor = the facing three-seed set S1 s16 / s17 / s18 through ``body_hold`` (stamps ``lr/ANC_S1_s<seed>/<row>/
seed00<es>/``). Per anchor and statistic, the read is the mean of its es1 and es2 reads (field_hard: completion
pooled over es1-es4, n = 400); ``band`` = that anchor's |es1 - es2|, floored (ratios 0.02, yaw rates 0.005 rad/s,
completion max(0.04, binomial SE)). "Worst" / "best" anchor = the weakest / strongest of the three on that statistic.
Body statistics over surviving episodes (bc_read). Gate texts: PLAN "Pre-registered gates".

STOP checks (PLAN step 4): an anchor-relative clause that would need a net ratio > 0.95 or a ripple < 0.03; the named
known-bad (P2 = S1 zero-shot in body mode) PASSING gate 2 or gate 4.

    bc_anchor_read.py [--json OUT] [--md OUT]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
C = HERE.parent
spec = importlib.util.spec_from_file_location("_bc_read_a", HERE / "bc_read.py")
BR = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BR)
LR = Path("/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/body_cmd/lr")
SEEDS = (16, 17, 18)
WZ0 = ("stand", "fwd", "lat_l", "lat_r", "back", "diag")
TRANSL = ("fwd", "lat_l", "lat_r", "back", "diag")
LEVEL_KEYS = ("w_min", "0p08", "0p15", "ceiling", "w_max")


def stamp(name: str, row: str, es: int) -> Path | None:
    d = LR / name / row / f"seed{es:03d}"
    hits = sorted(p for p in d.glob("*/") if (p / "run_meta.json").is_file()) if d.is_dir() else []
    return hits[-1] if hits else None


def level_labels(rows: dict) -> dict:
    """turnlevels labels -> level key, by the recorded wz (the symbols resolve to numbers)."""
    out = {}
    for lab in rows:
        if not lab.startswith("turn_"):
            continue
        for k in LEVEL_KEYS:
            if lab.endswith("_" + k):
                out[lab] = k
    return out


def pooled(rec_list, key, how="median"):
    vals = []
    for rec in rec_list:
        vals += [b.get(key) for b in rec["eps"] if b.get(key) is not None and math.isfinite(b.get(key))]
    vals = np.array(vals, dtype=float)
    if not vals.size:
        return None
    return float({"median": np.median, "mean": np.mean}[how](vals)) if how != "p90" else float(np.quantile(vals, .9))


def drift_stats(rows: dict) -> tuple[float | None, float | None]:
    vals = []
    for lab in WZ0:
        if lab in rows:
            vals += [b["heading_drift"] for b in rows[lab]["eps"] if b.get("heading_drift") is not None]
    v = np.array(vals, dtype=float)
    return (float(np.median(v)), float(np.quantile(v, 0.9))) if v.size else (None, None)


def per_anchor(name: str) -> dict:
    out = {"suite": {}, "levels": {}, "drift": {}, "field": {}}
    reads = {es: {"suite": BR.load(stamp(name, "suite__omni_body", es)),
                  "levels": BR.load(stamp(name, "turnlevels__omni_body", es))} for es in (1, 2)}
    for es, r in reads.items():
        s = r["suite"]
        out["drift"][es] = drift_stats(s)
        for lab, rec in s.items():
            out["suite"].setdefault(lab, {})[es] = {"survival": rec["survival"],
                                                    "body_ratio": BR.stat(rec, "body_ratio"),
                                                    "net_yaw_ratio": BR.stat(rec, "net_yaw_ratio"),
                                                    "yaw_ripple": BR.stat(rec, "yaw_ripple")}
        lv = r["levels"]
        for lab, key in level_labels(lv).items():
            d = out["levels"].setdefault(key, {}).setdefault(es, {"recs": []})
            d["recs"].append(lv[lab])
        for key, d in out["levels"].items():
            if es in d:
                recs = d[es]["recs"]
                d[es] = {"net_yaw_ratio": pooled(recs, "net_yaw_ratio"), "yaw_ripple": pooled(recs, "yaw_ripple"),
                         "net_yaw_rate_abs": float(np.median(np.abs([b["net_yaw_rate"] for rc in recs for b in rc["eps"]
                                                                     if b.get("net_yaw_rate") is not None]))),
                         "survival": float(np.mean([rc["survival"] for rc in recs]))}
    for row, ess in (("field_hard__omni_body", (1, 2, 3, 4)), ("field_light__omni_body", (1, 2)),
                     ("ring_obst__omni_body", (1, 2))):
        comp = {es: BR.completion(stamp(name, row, es)) for es in ess}
        out["field"][row] = {es: c[0] for es, c in comp.items()}
        out["field"][row]["n"] = sum((c[1] or 0) for c in comp.values())
    return out


def mean2(d: dict, key: str):
    v = [d[es][key] for es in (1, 2) if es in d and d[es].get(key) is not None]
    return float(np.mean(v)) if v else None


def band(d: dict, key: str, floor: float):
    v = [d[es][key] for es in (1, 2) if es in d and d[es].get(key) is not None]
    return max(floor, abs(v[0] - v[1])) if len(v) == 2 else floor


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(C / "anchor_numbers.json"))
    ap.add_argument("--md", default=str(C / "anchor_numbers.md"))
    a = ap.parse_args()
    A = {s: per_anchor(f"ANC_S1_s{s}") for s in SEEDS}
    G, stops = {}, []
    # gate 1: move_turn survival per hand >= worst anchor - 0.05
    for lab in ("move_turn_l", "move_turn_r"):
        worst = min(mean2(A[s]["suite"][lab], "survival") for s in SEEDS)
        G[f"g1.{lab}.survival_min"] = worst - 0.05
    # gate 2: ripple below the ceiling <= 0.5 x worst anchor; top rate at w_max >= best anchor - band
    for key in ("w_min", "0p08", "0p15"):
        rip = [mean2(A[s]["levels"][key], "yaw_ripple") for s in SEEDS]
        G[f"g2.{key}.ripple_max"] = 0.5 * max(rip)
        if 0.5 * max(rip) < 0.03:
            stops.append(f"gate 2 ripple clause at {key} would need ripple < 0.03 ({0.5 * max(rip):.4f})")
    tops = {s: mean2(A[s]["levels"]["w_max"], "net_yaw_rate_abs") for s in SEEDS}
    best = max(tops, key=tops.get)
    G["g2.w_max.net_rate_min"] = tops[best] - band(A[best]["levels"]["w_max"], "net_yaw_rate_abs", 0.005)
    # gate 3: move_turn body-frame translation |ratio - 1| <= the best anchor's
    for lab in ("move_turn_l", "move_turn_r"):
        devs = [abs(mean2(A[s]["suite"][lab], "body_ratio") - 1.0) for s in SEEDS]
        G[f"g3.{lab}.transl_dev_max"] = min(devs)
    # gate 4: drift
    med = [float(np.mean([A[s]["drift"][es][0] for es in (1, 2)])) for s in SEEDS]
    p90 = [float(np.mean([A[s]["drift"][es][1] for es in (1, 2)])) for s in SEEDS]
    G["g4.drift_median_max"] = max(0.01, 2 * max(med))
    G["g4.drift_p90_max"] = max(0.02, 2 * max(p90))
    # gate 5: category |ratio - 1| <= worst anchor + max(0.03, band); fields >= worst anchor - max(floor, 2 band)
    for lab in TRANSL:
        devs = {s: abs(mean2(A[s]["suite"][lab], "body_ratio") - 1.0) for s in SEEDS}
        w = max(devs, key=devs.get)
        G[f"g5.{lab}.ratio_dev_max"] = devs[w] + max(0.03, band(A[w]["suite"][lab], "body_ratio", 0.02))
    for row, floor in (("field_hard__omni_body", 0.05), ("field_light__omni_body", 0.05), ("ring_obst__omni_body", 0.05)):
        comps = {s: float(np.mean([v for k, v in A[s]["field"][row].items() if k != "n"])) for s in SEEDS}
        w = min(comps, key=comps.get)
        n = A[w]["field"][row]["n"]
        p = comps[w]
        es_vals = [A[w]["field"][row][es] for es in (1, 2)]
        se = math.sqrt(max(p * (1 - p), 1e-9) / max(n, 1))
        bnd = max(0.04, se, abs(es_vals[0] - es_vals[1]))
        G[f"g5.{row}.completion_min"] = p - max(floor, 2 * bnd)
    # "an anchor-relative clause would need a net ratio > 0.95": gate 3's / gate 2's ratio floors are absolute
    # (0.8 / 0.5) -- only flag if the anchors themselves imply a > 0.95 ratio requirement (none by construction).
    # P2 known-bad: must FAIL gate 2 and gate 4
    p2s, p2l = BR.load(stamp("P2_S1_s18", "suite__omni_body", 1)), BR.load(stamp("P2_S1_s18", "turnlevels__omni_body", 1))
    p2_med, p2_p90 = drift_stats(p2s)
    p2_g4_pass = p2_med is not None and p2_med <= G["g4.drift_median_max"] and p2_p90 <= G["g4.drift_p90_max"]
    lv = {}
    for lab, key in level_labels(p2l).items():
        lv.setdefault(key, []).append(p2l[lab])
    p2_ratio = {k: pooled(v, "net_yaw_ratio") for k, v in lv.items()}
    p2_g2_pass = (all((p2_ratio.get(k) or -9) >= 0.8 for k in ("0p08", "0p15", "ceiling"))
                  and (p2_ratio.get("w_min") or -9) >= 0.5)
    if p2_g4_pass:
        stops.append(f"known-bad P2 PASSES gate 4 (drift median {p2_med:.4f}, p90 {p2_p90:.4f})")
    if p2_g2_pass:
        stops.append(f"known-bad P2 PASSES gate 2's ratio clauses ({p2_ratio})")
    res = {"anchors": {s: {"drift": A[s]["drift"], "levels": {k: {es: {kk: vv for kk, vv in d.items() if kk != 'recs'}
                                                                       for es, d in v.items()}
                                                                   for k, v in A[s]["levels"].items()},
                           "suite": A[s]["suite"], "field": A[s]["field"]} for s in SEEDS},
           "gate_numbers": G, "p2": {"drift_median": p2_med, "drift_p90": p2_p90, "level_ratio": p2_ratio,
                                     "passes_g2": p2_g2_pass, "passes_g4": p2_g4_pass},
           "stops": stops}
    Path(a.json).write_text(json.dumps(res, indent=2, default=str) + "\n")
    L = ["# Step 4 — anchor-relative gate numbers (frozen into the CHARTER)", "",
         "Anchor = S1 s16 / s17 / s18 + `body_hold`, es1 + es2 (field_hard es1–es4). Rule: PLAN gates, CHARTER.", "",
         "| clause | number |", "|---|---|"]
    L += [f"| {k} | {v:.4f} |" for k, v in G.items()]
    L += ["", f"P2 (S1 zero-shot, body mode): drift median {p2_med}, p90 {p2_p90}; level net ratios {p2_ratio}; "
          f"passes gate 2: {p2_g2_pass}; passes gate 4: {p2_g4_pass}", "", f"**STOPS:** {stops or 'none'}", "",
          "| anchor | level | net ratio (es1, es2) | ripple | net rate | survival |", "|---|---|---|---|---|---|"]
    for s in SEEDS:
        for k in LEVEL_KEYS:
            d = A[s]["levels"].get(k, {})
            fm = lambda key: ", ".join(f"{d[es][key]:.3f}" if d.get(es, {}).get(key) is not None else "—" for es in (1, 2))
            L.append(f"| s{s} | {k} | {fm('net_yaw_ratio')} | {fm('yaw_ripple')} | {fm('net_yaw_rate_abs')} | {fm('survival')} |")
    Path(a.md).write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 1 if stops else 0


if __name__ == "__main__":
    raise SystemExit(main())
