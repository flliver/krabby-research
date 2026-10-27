#!/usr/bin/env python3
"""Step 6 stage-end read (CPU): a body stage head vs its three same-stage anchors, on ``suite__omni_body`` es1.

PLAN step 6 (as amended by the pass-3 audit fix, CHARTER):
* on the stage's TRAINED categories, a shortfall against the WORST same-stage anchor beyond max(floor, the three
  anchors' range) is decide-record-notify -> verdict NOTIFY (floors: |speed ratio - 1| 0.02, survival 0.04);
* STOP and ask only on a COLLAPSE -- survival or speed ratio < 0.5 x the worst same-stage anchor's -- or when heading
  drift fails gate 4's clauses against the same-stage anchors (median |mean wz| on the wz = 0 holds <= max(0.01,
  2 x the worst anchor's median), p90 <= max(0.02, 2 x the worst anchor's p90); pooled over the stage's trained
  wz = 0 holds);
* turn rows are descriptive; ``--low-end STAMP`` adds the c6_b check: the pure-turn net ratio at w_min (turnlevels
  row, both hands) < 0.5 -> STOP.
Body statistics over surviving episodes only (bc_read).

    bc_stage_read.py --stage NAME --head STAMP --anchor STAMP --anchor STAMP --anchor STAMP --categories a,b,..
                     [--low-end TURNLEVELS_STAMP] [--json OUT]
Exit 0 OK, 2 NOTIFY, 3 STOP.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_bc_read_s", HERE / "bc_read.py")
BR = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BR)

LABELS = {"stand": ("stand",), "fwd": ("fwd",), "side": ("lat_l", "lat_r"), "back": ("back",), "diag": ("diag",),
          "turn": ("turn_l_ceiling", "turn_r_ceiling"), "move_turn": ("move_turn_l", "move_turn_r")}
TRANSLATE = ("fwd", "side", "back", "diag")
RATIO_FLOOR, SURV_FLOOR = 0.02, 0.04


def drift_pool(rows: dict, labels) -> np.ndarray:
    vals = []
    for lab in labels:
        rec = rows.get(lab)
        if rec:
            vals += [b.get("heading_drift") for b in rec["eps"] if b.get("heading_drift") is not None]
    return np.array(vals, dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--anchor", action="append", required=True)
    ap.add_argument("--categories", required=True)
    ap.add_argument("--low-end", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    cats = [c for c in a.categories.split(",") if c]
    head = BR.load(a.head)
    anchors = [BR.load(s) for s in a.anchor]
    notify, stop, table = [], [], []
    for c in cats:
        if c not in TRANSLATE and c != "stand":
            continue
        for lab in LABELS[c]:
            h = head.get(lab)
            an = [x.get(lab) for x in anchors]
            if not h or any(x is None for x in an):
                notify.append(f"{lab}: missing in head or an anchor")
                continue
            hs, asv = h["survival"], [x["survival"] for x in an]
            row = {"label": lab, "survival": hs, "anchor_survival": asv}
            worst_s, rng_s = min(asv), max(asv) - min(asv)
            if hs < 0.5 * worst_s:
                stop.append(f"{lab}: survival {hs:.3f} < 0.5 x worst anchor {worst_s:.3f} (COLLAPSE)")
            elif hs < worst_s - max(SURV_FLOOR, rng_s):
                notify.append(f"{lab}: survival {hs:.3f} < worst anchor {worst_s:.3f} - max({SURV_FLOOR}, range "
                              f"{rng_s:.3f})")
            if c in TRANSLATE:
                hr = BR.stat(h, "body_ratio")
                ar = [BR.stat(x, "body_ratio") for x in an]
                row.update({"body_ratio": hr, "anchor_body_ratio": ar})
                if hr is None or any(x is None for x in ar):
                    notify.append(f"{lab}: body ratio missing")
                else:
                    worst_r = min(ar)
                    dev, adev = abs(hr - 1.0), [abs(x - 1.0) for x in ar]
                    if hr < 0.5 * worst_r:
                        stop.append(f"{lab}: speed ratio {hr:.3f} < 0.5 x worst anchor {worst_r:.3f} (COLLAPSE)")
                    elif dev > max(adev) + max(RATIO_FLOOR, max(adev) - min(adev)):
                        notify.append(f"{lab}: |ratio - 1| {dev:.3f} > worst anchor {max(adev):.3f} + max("
                                      f"{RATIO_FLOOR}, range {max(adev) - min(adev):.3f})")
            table.append(row)
    # drift (gate 4 clauses vs the same-stage anchors), pooled over the stage's trained wz = 0 holds
    wz0 = [lab for c in cats if c in TRANSLATE or c == "stand" for lab in LABELS[c]]
    hd = drift_pool(head, wz0)
    ad = [drift_pool(x, wz0) for x in anchors]
    drift = {"labels": wz0, "head_median": float(np.median(hd)) if hd.size else None,
             "head_p90": float(np.quantile(hd, 0.9)) if hd.size else None,
             "anchor_median": [float(np.median(x)) if x.size else None for x in ad],
             "anchor_p90": [float(np.quantile(x, 0.9)) if x.size else None for x in ad]}
    if hd.size and all(x.size for x in ad):
        lim_m = max(0.01, 2 * max(drift["anchor_median"]))
        lim_p = max(0.02, 2 * max(drift["anchor_p90"]))
        drift.update({"limit_median": lim_m, "limit_p90": lim_p})
        if drift["head_median"] > lim_m:
            stop.append(f"drift: median {drift['head_median']:.4f} > {lim_m:.4f} (gate 4 vs the same-stage anchors)")
        if drift["head_p90"] > lim_p:
            stop.append(f"drift: p90 {drift['head_p90']:.4f} > {lim_p:.4f} (gate 4 vs the same-stage anchors)")
    else:
        notify.append("drift: no wz = 0 holds read")
    turns = {lab: {"survival": head[lab]["survival"], "net_yaw_ratio": BR.stat(head[lab], "net_yaw_ratio"),
                   "yaw_ripple": BR.stat(head[lab], "yaw_ripple"), "body_ratio": BR.stat(head[lab], "body_ratio")}
             for c in ("turn", "move_turn") for lab in LABELS[c] if lab in head}
    low = None
    if a.low_end:
        lr = BR.load(a.low_end)
        low = {lab: BR.stat(rec, "net_yaw_ratio") for lab, rec in lr.items() if "w_min" in lab}
        for lab, v in low.items():
            if v is None or v < 0.5:
                stop.append(f"low end: {lab} net ratio {v} < 0.5 (c6_b check)")
    verdict = "STOP" if stop else ("NOTIFY" if notify else "OK")
    res = {"stage": a.stage, "head": a.head, "anchors": a.anchor, "categories": cats, "verdict": verdict,
           "stop": stop, "notify": notify, "table": table, "drift": drift, "turns_descriptive": turns,
           "low_end": low}
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({k: res[k] for k in ("stage", "verdict", "stop", "notify", "drift", "turns_descriptive",
                                           "low_end")}, indent=1))
    return {"OK": 0, "NOTIFY": 2, "STOP": 3}[verdict]


if __name__ == "__main__":
    raise SystemExit(main())
