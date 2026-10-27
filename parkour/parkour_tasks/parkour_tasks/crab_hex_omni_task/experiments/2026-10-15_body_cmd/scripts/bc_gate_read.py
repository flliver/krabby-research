#!/usr/bin/env python3
"""PLAN step 7 / 8 gate read (CPU): one body head's gate rows vs the frozen anchor numbers (gates 0-5, gate 7's
per-head clause). Gate 6 (stand pair) is step 9's.

Inputs: ``--name GATE_s18`` (the head's stamps under lr/<name>/), ``anchor_numbers.json`` (step 4) and the anchors'
own stamps (for the per-level ripple / hand references). Statistics over surviving episodes (bc_read); a head's
read = the mean of its es1 / es2 reads (field_hard completion pooled over es1-es4).

Three clauses depend on the user's step-4 decision (CHARTER 2026-10-16); until it is recorded the reader applies
the RECOMMENDED forms and marks them PENDING in the output:
  --ripple {best,worst,off}   gate 2 ripple below the ceiling <= the best (smoothest) / worst anchor's + 0.005, or
                              not gated (default best)
  --g3-transl {best_band,best} gate 3 translation |ratio - 1| <= the best anchor's + max(0.02, band), or bare (best)
  --pending / --final          stamp the output as pending or final

    bc_gate_read.py --name GATE_s18 [--ripple best] [--g3-transl best_band] [--pending|--final] [--json OUT]
Exit 0 all of gates 0-5 + gate 7 PASS, 1 otherwise.
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


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BR = _load("_bc_read_g", HERE / "bc_read.py")
AR = _load("_bc_anchor_g", HERE / "bc_anchor_read.py")
SM = _load("_bc_smoke_g", HERE / "bc_smoke_check.py")
FINAL_CODE, FINAL_EVAL = "fe714b41490619fd", "567d17040a181542"
LEVELS = ("w_min", "0p08", "0p15", "ceiling", "w_max")
STRAIGHT = ("stand", "fwd", "lat_l", "lat_r", "back", "diag")
TURN_SUITE = ("turn_l_ceiling", "turn_r_ceiling")


def stamps(name, row, ess):
    return {es: AR.stamp(name, row, es) for es in ess}


def gate0(name: str) -> list[str]:
    bad = []
    for row, ess in (("suite__omni_body", (1, 2)), ("turnlevels__omni_body", (1, 2)),
                     ("field_hard__omni_body", (1, 2, 3, 4)), ("field_light__omni_body", (1, 2)),
                     ("ring_obst__omni_body", (1, 2))):
        for es, st in stamps(name, row, ess).items():
            if st is None:
                bad.append(f"{row} es{es}: missing")
                continue
            m = json.loads((st / "run_meta.json").read_text())
            ev = m.get("env_vars") or {}
            cs = json.loads((st / "code_state.json").read_text())
            if not cs["code_sha"].startswith(FINAL_CODE) or not cs["eval_sha"].startswith(FINAL_EVAL) or \
                    cs.get("dirty") != "False":
                bad.append(f"{row} es{es}: code/eval sha {cs['code_sha'][:12]}/{cs['eval_sha'][:12]} dirty {cs['dirty']}")
            for k, v in (("KRABBY_OMNI_COMMAND_FRAME", "body"), ("KRABBY_STD_PARAM", "soft"),
                         ("KRABBY_OMNI_SPEED_ELLIPSE", "0.35:0.35:0.25"), ("KRABBY_OMNI_FACING_OBS_ENCODING", "none"),
                         ("KRABBY_OMNI_BODY_ENVELOPE", "0.05:0.168:0.21")):
                if ev.get(k) != v:
                    bad.append(f"{row} es{es}: {k}={ev.get(k)!r} (want {v!r})")
            if "KRABBY_OMNI_CLOCK_HIDE" not in ev:
                bad.append(f"{row} es{es}: CLOCK_HIDE not recorded")
            if row in ("suite__omni_body", "turnlevels__omni_body"):
                t = SM.check_term(st)
                if not t["pass"]:
                    bad.append(f"{row} es{es}: column / command checks {t['bad']}")
    return bad


def m2(d: dict, key: str):
    """Mean of an anchor read's es1 / es2 values (anchor_numbers.json keys the env seeds as strings)."""
    v = [d[es][key] for es in ("1", "2") if es in d and d[es].get(key) is not None]
    return float(np.mean(v)) if v else None


def mean_es(d):
    v = [x for x in d.values() if x is not None]
    return float(np.mean(v)) if v else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ripple", choices=("best", "worst", "off"), default="best")
    ap.add_argument("--g3-transl", choices=("best_band", "best"), default="best_band")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--pending", action="store_true", default=True)
    g.add_argument("--final", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    N = json.loads((C / "anchor_numbers.json").read_text())
    G = N["gate_numbers"]
    anc = N["anchors"]
    res = {"name": a.name, "status": "FINAL" if a.final else "PENDING (step-4 gate decision not recorded)",
           "options": {"ripple": a.ripple, "g3_transl": a.g3_transl}, "gates": {}}
    # gate 0
    g0 = gate0(a.name)
    res["gates"]["0_invalid"] = {"pass": not g0, "why": g0}
    # per-es reads
    suite = {es: BR.load(st) for es, st in stamps(a.name, "suite__omni_body", (1, 2)).items() if st}
    lev = {es: BR.load(st) for es, st in stamps(a.name, "turnlevels__omni_body", (1, 2)).items() if st}
    # gate 1
    why = []
    for lab in STRAIGHT + TURN_SUITE:
        s = mean_es({es: suite[es][lab]["survival"] for es in suite if lab in suite[es]})
        if s is None or s < 0.95:
            why.append(f"{lab} survival {s}")
    for lab in [l for l in lev.get(1, {}) if l.startswith("turn_")]:
        s = mean_es({es: lev[es][lab]["survival"] for es in lev if lab in lev[es]})
        if s is None or s < 0.95:
            why.append(f"{lab} survival {s}")
    for lab in ("move_turn_l", "move_turn_r"):
        s = mean_es({es: suite[es][lab]["survival"] for es in suite})
        if s is None or s < G[f"g1.{lab}.survival_min"]:
            why.append(f"{lab} survival {s} < {G[f'g1.{lab}.survival_min']:.3f}")
    res["gates"]["1_floor"] = {"pass": not why, "why": why}
    # gate 2
    why, table = [], {}
    for hand in ("l", "r"):
        prev = None
        for k in LEVELS:
            lab = f"turn_{hand}_{k}"
            ratio = mean_es({es: BR.stat(lev[es][lab], "net_yaw_ratio") for es in lev})
            rate = mean_es({es: abs(BR.stat(lev[es][lab], "net_yaw_rate") or 0.0) for es in lev})
            table[lab] = {"net_yaw_ratio": ratio, "net_rate_abs": rate}
            if k == "w_min" and (ratio is None or ratio < 0.5):
                why.append(f"{lab} net ratio {ratio} < 0.5")
            if k in ("0p08", "0p15", "ceiling") and (ratio is None or ratio < 0.8):
                why.append(f"{lab} net ratio {ratio} < 0.8")
            if prev is not None and rate is not None and rate < prev - 0.005:
                why.append(f"{lab} not monotone: {rate:.4f} < previous {prev:.4f} - 0.005")
            prev = rate
    rip = {}
    for k in ("w_min", "0p08", "0p15"):
        recs = {es: [lev[es][f"turn_{h}_{k}"] for h in ("l", "r")] for es in lev}
        r = mean_es({es: AR.pooled(v, "yaw_ripple") for es, v in recs.items()})
        an = [m2(anc[s]["levels"][k], "yaw_ripple") for s in ("16", "17", "18")]
        lim = (min(an) if a.ripple == "best" else max(an)) + 0.005
        rip[k] = {"head": r, "limit": lim if a.ripple != "off" else None}
        if a.ripple != "off" and (r is None or r > lim):
            why.append(f"ripple at {k} {r} > {lim:.4f} ({a.ripple} anchor + 0.005)")
    top = mean_es({es: AR.pooled([lev[es][f"turn_{h}_w_max"] for h in ("l", "r")], "net_yaw_rate") for es in lev})
    top_abs = mean_es({es: float(np.median(np.abs([b["net_yaw_rate"] for h in ("l", "r")
                                                    for b in lev[es][f"turn_{h}_w_max"]["eps"]]))) for es in lev})
    if top_abs is None or top_abs < G["g2.w_max.net_rate_min"]:
        why.append(f"top rate at w_max {top_abs} < {G['g2.w_max.net_rate_min']:.4f}")
    res["gates"]["2_turn_works"] = {"pass": not why, "why": why, "levels": table, "ripple": rip, "top_rate": top_abs}
    # gate 3
    why, mt = [], {}
    for lab in ("move_turn_l", "move_turn_r"):
        ratio = mean_es({es: BR.stat(suite[es][lab], "net_yaw_ratio") for es in suite})
        br = mean_es({es: BR.stat(suite[es][lab], "body_ratio") for es in suite})
        dev = None if br is None else abs(br - 1.0)
        lim = G[f"g3.{lab}.transl_dev_max"]
        if a.g3_transl == "best_band":
            # the best anchor's own es1/es2 band, floored at 0.02
            best = min(("16", "17", "18"), key=lambda s: abs(m2(anc[s]["suite"][lab], "body_ratio") - 1.0))
            d = anc[best]["suite"][lab]
            b = abs(d["1"]["body_ratio"] - d["2"]["body_ratio"]) if "1" in d and "2" in d else 0.0
            lim = lim + max(0.02, b)
        mt[lab] = {"net_yaw_ratio": ratio, "body_ratio": br, "transl_dev": dev, "transl_limit": lim}
        if ratio is None or ratio < 0.8:
            why.append(f"{lab} net ratio {ratio} < 0.8")
        if dev is None or dev > lim:
            why.append(f"{lab} |ratio - 1| {dev} > {lim:.4f}")
    res["gates"]["3_move_turn"] = {"pass": not why, "why": why, "rows": mt}
    # gate 4
    vals = []
    for es in suite:
        for lab in STRAIGHT:
            vals += [b["heading_drift"] for b in suite[es][lab]["eps"] if b.get("heading_drift") is not None]
    v = np.array(vals, dtype=float)
    med, p90 = (float(np.median(v)), float(np.quantile(v, 0.9))) if v.size else (None, None)
    why = []
    if med is None or med > G["g4.drift_median_max"]:
        why.append(f"drift median {med} > {G['g4.drift_median_max']:.4f}")
    if p90 is None or p90 > G["g4.drift_p90_max"]:
        why.append(f"drift p90 {p90} > {G['g4.drift_p90_max']:.4f}")
    res["gates"]["4_straight"] = {"pass": not why, "why": why, "median": med, "p90": p90}
    # gate 5
    why, rr = [], {}
    for lab in ("fwd", "lat_l", "lat_r", "back", "diag"):
        br = mean_es({es: BR.stat(suite[es][lab], "body_ratio") for es in suite})
        dev = None if br is None else abs(br - 1.0)
        rr[lab] = {"body_ratio": br, "dev": dev, "limit": G[f"g5.{lab}.ratio_dev_max"]}
        if dev is None or dev > G[f"g5.{lab}.ratio_dev_max"]:
            why.append(f"{lab} |ratio - 1| {dev} > {G[f'g5.{lab}.ratio_dev_max']:.4f}")
    for row, ess in (("field_hard__omni_body", (1, 2, 3, 4)), ("field_light__omni_body", (1, 2)),
                     ("ring_obst__omni_body", (1, 2))):
        comps = [BR.completion(st)[0] for st in stamps(a.name, row, ess).values() if st]
        comp = float(np.mean(comps)) if comps else None
        rr[row] = {"completion": comp, "limit": G[f"g5.{row}.completion_min"]}
        if comp is None or comp < G[f"g5.{row}.completion_min"]:
            why.append(f"{row} completion {comp} < {G[f'g5.{row}.completion_min']:.3f}")
    res["gates"]["5_retention"] = {"pass": not why, "why": why, "rows": rr}
    # gate 7 per-head clause: no hand loss (a side hand slower than the worst anchor's by > band)
    why = []
    for lab in ("lat_l", "lat_r"):
        an = {s: m2(anc[s]["suite"][lab], "body_ratio") for s in ("16", "17", "18")}
        w = min(an, key=an.get)
        d = anc[w]["suite"][lab]
        bnd = max(0.02, abs(d["1"]["body_ratio"] - d["2"]["body_ratio"]))
        hr = rr[lab]["body_ratio"]
        if hr is None or hr < an[w] - bnd:
            why.append(f"{lab} body ratio {hr} < worst anchor {an[w]:.3f} - band {bnd:.3f} (hand loss)")
    res["gates"]["7_head_clause"] = {"pass": not why, "why": why}
    res["pass"] = all(v["pass"] for v in res["gates"].values())
    out = Path(a.json) if a.json else C / f"gates_{a.name}.json"
    out.write_text(json.dumps(res, indent=2, default=str) + "\n")
    print(json.dumps({k: (v["pass"], v["why"]) for k, v in res["gates"].items()}, indent=1))
    print("ALL PASS" if res["pass"] else "FAIL", "|", res["status"])
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
