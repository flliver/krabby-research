#!/usr/bin/env python3
"""Step 1a sensitivity (CPU): the w_min share under window length, sigma, tent weight and w_min variants.

Same traces and income definition as bc_turn_pricing.py; only the knobs move. Writes step1a_sweep.json/.md.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_bc_tp", HERE / "bc_turn_pricing.py")
TP = importlib.util.module_from_spec(spec)
spec.loader.exec_module(TP)


def shares(wob, dt, c, win_s, sigma, w_track, w_tent):
    n = int(round(win_s / dt))
    dy = np.concatenate([TP.window_sum(w, n, dt) for w in wob])
    def inc(d):
        p = math.copysign(1.0, c) * d / win_s
        return w_track * np.exp(-np.abs(d - c * win_s) / sigma) + w_tent * np.minimum(p, 2 * abs(c) - p)
    trk = inc(dy + c * win_s).mean()
    sip = inc(dy).mean()
    still = float(inc(np.zeros(1)).mean())
    return {"tracker": float(trk), "sip": float(sip / trk), "still": float(still / trk),
            "premium_per_s": float((trk - still) / dt)}


def main():
    traces, dt = TP.hold_traces(TP.STAMP)
    fams = {"translation": TP.TRANSLATE, "move_turn": TP.MOVE_TURN}
    wob = {f: [w - w.mean() for lab in labs for w in traces[lab]] for f, labs in fams.items()}
    variants = [
        ("as planned: 1 s, sigma 0.05, tent 1.0, c 0.05", 1.0, 0.05, 0.2, 1.0, 0.05),
        ("w_min 0.06", 1.0, 0.05, 0.2, 1.0, 0.06),
        ("w_min 0.07", 1.0, 0.05, 0.2, 1.0, 0.07),
        ("2 s window, sigma 0.10", 2.0, 0.10, 0.2, 1.0, 0.05),
        ("2 s window, sigma 0.10, at 0.08", 2.0, 0.10, 0.2, 1.0, 0.08),
        ("2 s window, sigma 0.10, at ceiling", 2.0, 0.10, 0.2, 1.0, 0.167),
        ("1 s, sigma 0.03", 1.0, 0.03, 0.2, 1.0, 0.05),
        ("1 s, tent weight 2.0", 1.0, 0.05, 0.2, 2.0, 0.05),
        ("2 s window, sigma 0.10, tent 2.0", 2.0, 0.10, 0.2, 2.0, 0.05),
    ]
    out = []
    for name, win, sig, wt, wtent, c in variants:
        row = {"variant": name, "win_s": win, "sigma": sig, "w_track": wt, "w_tent": wtent, "c": c}
        for f in fams:
            row[f] = shares(wob[f], dt, c, win, sig, wt, wtent)
        out.append(row)
    (HERE.parent / "step1a_sweep.json").write_text(json.dumps(out, indent=2) + "\n")
    L = ["# Step 1a sensitivity — non-rotating share of the yaw-term income", "",
         "Share = non-rotating trace income / healthy tracker income (lower = slow turns better priced). "
         "Same S1 traces as step1a_turn_pricing.md.", "",
         "| variant | c | transl. share still | transl. share SIP | move_turn share still | move_turn share SIP | "
         "transl. premium /s |", "|---|---|---|---|---|---|---|"]
    for r in out:
        t, m = r["translation"], r["move_turn"]
        L.append(f"| {r['variant']} | {r['c']:.3f} | {t['still']:.3f} | {t['sip']:.3f} | {m['still']:.3f} | "
                 f"{m['sip']:.3f} | {t['premium_per_s']:.2f} |")
    (HERE.parent / "step1a_sweep.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
