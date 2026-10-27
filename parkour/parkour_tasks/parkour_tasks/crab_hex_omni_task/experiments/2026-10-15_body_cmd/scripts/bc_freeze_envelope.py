#!/usr/bin/env python3
"""Step 4 (CPU): freeze the command envelope from P1, by the rule pinned in CHARTER §3 (2026-10-15).

* ceiling = mean over both hands x es1 + es2 of S1 s18's net heading change over the first 6 s of the WORLD suite's
  pure-turn holds / 6 s (REFS_S1_s18 suite stamps -- reproduced bitwise at the new code by G0), rounded to 0.001;
* w_max = 1.2 x ceiling rounded UP to 0.01;
* cap = (0, w_max) + for |v| in {0.22, 0.30}: 1.2 x min(fwd, lat) achieved net yaw rate at wz cmd 0.30 (P1 grid via
  body_hold; survivor median per cell, mean of es1 / es2), rounded DOWN to 0.005, non-increasing, <= w_max;
* cross-check: the spot cell (|v| 0, wz cmd 0.30) net rate vs the ceiling -- STOP if they differ by > 0.01;
* STOP if cap(0.22) < w_min (no move+turn pair).
Also prints the whole P1 map (every cell: achieved net rate, ratio, survival, body ratio).

    bc_freeze_envelope.py --p1 STAMP_ES1 --p1 STAMP_ES2 [--json OUT] [--md OUT]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_bc_read", HERE / "bc_read.py")
BR = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BR)

R = Path("/home/nickmagus/krabby/krabby-research")
REFS = R / "parkour/outputs/omni_plan/phase2/lr/REFS_S1_s18/suite__omni"
REFS_STAMPS = (REFS / "seed001/2026-10-09_03-20-24", REFS / "seed002/2026-10-09_03-27-23")
W_MIN = 0.05
HEADROOM = 1.2
CAP_V = (0.22, 0.30)
SAT_WZ = 0.30


def world_ceiling() -> tuple[float, list]:
    reads = []
    for st in REFS_STAMPS:
        m = json.loads((st / "run_meta.json").read_text())
        dt = float(m["dt"])
        n = int(round(6.0 / dt))
        for lab, sg in (("turn_l90", 1.0), ("turn_r90", -1.0)):
            vals = []
            for row in m["env_index_map"]:
                if row["label"] != lab:
                    continue
                z = np.load(st / "raw" / f"episode_{row['env_index']:02d}.npz")
                h = z["segment_id"] >= 1
                q = z["root_quat_w"][h]
                yaw = np.unwrap(np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2)))
                vals.append(sg * (yaw[n] - yaw[0]) / 6.0)
            reads.append({"stamp": st.name, "hold": lab, "net6": float(np.mean(vals)), "n": len(vals)})
    return float(np.mean([r["net6"] for r in reads])), reads


def cell_key(h: dict) -> tuple[float, str, float]:
    v = math.hypot(h["vx"], h["vy"])
    axis = "spot" if v == 0 else ("fwd" if abs(h["vy"]) < 1e-9 else "lat")
    return (round(v, 3), axis, round(h["wz"], 3))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1", action="append", required=True)
    ap.add_argument("--json", default=str(HERE.parent / "envelope_freeze.json"))
    ap.add_argument("--md", default=str(HERE.parent / "envelope_freeze.md"))
    a = ap.parse_args()
    ceiling_raw, reads = world_ceiling()
    ceiling = round(ceiling_raw, 3)
    w_max = math.ceil(HEADROOM * ceiling * 100 - 1e-9) / 100
    cells: dict = {}
    for st in a.p1:
        cmds = BR.hold_cmds(st)
        rows = BR.load(st)
        for lab, rec in rows.items():
            if lab not in cmds:
                continue
            key = cell_key(cmds[lab])
            c = cells.setdefault(key, {"label": lab, "net": [], "ratio": [], "surv": [], "body": []})
            c["net"].append(BR.stat(rec, "net_yaw_rate"))
            c["ratio"].append(BR.stat(rec, "net_yaw_ratio"))
            c["surv"].append(rec["survival"])
            c["body"].append(BR.stat(rec, "body_ratio"))
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return float(np.mean(xs)) if xs else None
    grid = {f"{k[0]}|{k[1]}|{k[2]}": {"label": c["label"], "net_yaw_rate": mean(c["net"]),
                                      "net_yaw_ratio": mean(c["ratio"]), "survival": mean(c["surv"]),
                                      "body_ratio": mean(c["body"]), "n_reads": len(c["net"])}
            for k, c in sorted(cells.items())}
    stops = []
    spot = grid.get(f"0.0|spot|{SAT_WZ}", {}).get("net_yaw_rate")
    if spot is None or abs(spot - ceiling) > 0.01:
        stops.append(f"cross-check: spot cell net rate {spot} vs ceiling {ceiling} differ by > 0.01 (or missing)")
    cap = [(0.0, w_max)]
    for v in CAP_V:
        f = grid.get(f"{v}|fwd|{SAT_WZ}", {}).get("net_yaw_rate")
        l = grid.get(f"{v}|lat|{SAT_WZ}", {}).get("net_yaw_rate")
        if f is None or l is None:
            stops.append(f"cap: cell |v| {v} missing (fwd {f}, lat {l})")
            continue
        wz = math.floor(HEADROOM * min(f, l) / 0.005 + 1e-9) * 0.005
        wz = min(wz, cap[-1][1], w_max)
        cap.append((v, round(wz, 3)))
    if len(cap) > 1 and cap[1][1] < W_MIN:
        stops.append(f"cap(0.22) = {cap[1][1]} < w_min {W_MIN}: no move+turn pair (PLAN D3 STOP)")
    env = {"w_min": W_MIN, "ceiling": ceiling, "w_max": w_max, "cap": [[v, wz] for v, wz in cap]}
    res = {"ceiling_raw": ceiling_raw, "ceiling_reads": reads, "spot_cell_net": spot, "envelope": env, "grid": grid,
           "stops": stops, "knobs": {"KRABBY_OMNI_BODY_ENVELOPE": f"{W_MIN}:{ceiling}:{w_max}",
                                     "KRABBY_OMNI_BODY_CAP": ",".join(f"{v:g}@{wz:g}" for v, wz in cap)}}
    Path(a.json).write_text(json.dumps(res, indent=2) + "\n")
    L = ["# Step 4 — envelope freeze (P1)", "", f"Rule: CHARTER §3 2026-10-15. P1 stamps: {', '.join(a.p1)}", "",
         f"- ceiling (world suite first-6-s net, 4 reads) = {ceiling_raw:.4f} → **{ceiling}**; w_max = **{w_max}**",
         f"- spot cell (|v| 0, wz 0.30, body_hold) net = {spot}", f"- cap = **{cap}**",
         f"- knobs: `{res['knobs']['KRABBY_OMNI_BODY_ENVELOPE']}` / `{res['knobs']['KRABBY_OMNI_BODY_CAP']}`",
         f"- STOPS: {stops or 'none'}", "", "| |v| | axis | wz cmd | net yaw rate | net ratio | survival | body ratio |",
         "|---|---|---|---|---|---|---|"]
    for k, g in grid.items():
        v, ax, wz = k.split("|")
        fm = lambda x: "—" if x is None else f"{x:.3f}"
        L.append(f"| {v} | {ax} | {wz} | {fm(g['net_yaw_rate'])} | {fm(g['net_yaw_ratio'])} | {fm(g['survival'])} | "
                 f"{fm(g['body_ratio'])} |")
    Path(a.md).write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 1 if stops else 0


if __name__ == "__main__":
    raise SystemExit(main())
