#!/usr/bin/env python3
"""Step 5 smoke assertions (CPU) on the smoke heads' eval stamps (PLAN step 5).

Per ``--term STAMP`` (a body head on body rows, route term):
  obs      obs cols 6-7 exactly 0; col 5 == cmd wz; cols 8, 9 == cmd vx, vy; col 10 == |cmd xy|
  command  cmd == the hold row (vx, vy, wz), bitwise, on every live hold step
  clock    on every turning hold (wz != 0, down to w_min): clock_running on every live hold step after the first
           second, clock_cmd_speed >= the walk floor 0.21, and the clock phase advancing (>= 99 % of steps)
Per ``--world-rate STAMP`` (a body head on world rows through world_rate):
  obs cols 6-7 == 0; cols 5 / 8 / 9 / 10 == the adapter's applied command; applied wz in {0} U [w_min, w_max];
  route metadata world_rate / body; the clipped-step share by hold is reported.
The dump audits ran in bc_after_stage.py --smoke (stage_reads/<chain>.json); they are re-read and required PASS.

    bc_smoke_check.py --term STAMP [...] --world-rate STAMP [...] --chain BC_SMK_x_s18 [...] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
C = HERE.parent
TOL = 1e-6
FLOOR = 0.21


def check_term(stamp: Path) -> dict:
    m = json.loads((stamp / "run_meta.json").read_text())
    rows = {h["label"]: h for h in m["hold_body_cmds"]}
    labels = m["hold_labels"]
    bad = {"obs_facing": 0, "obs_cmd": 0, "cmd_row": 0, "clock_running": 0, "clock_speed": 0, "clock_still": 0}
    steps = turn_steps = 0
    route = (m.get("command_route"), m.get("command_adapter"), m.get("env_command_frame"))
    dt = float(m["dt"])
    for f in sorted((stamp / "raw").glob("episode_*.npz")):
        z = np.load(f)
        seg = z["segment_id"].astype(int)
        live = (seg >= 0) & ~z["done"]
        obs, cmd = z["obs_cmd"], z["cmd_applied"]
        idx = np.where(live)[0]
        steps += idx.size
        bad["obs_facing"] += int((np.abs(obs[idx, 1:3]) > 0).any(axis=1).sum())
        e = np.max(np.abs(np.stack([obs[idx, 0] - cmd[idx, 2], obs[idx, 3] - cmd[idx, 0], obs[idx, 4] - cmd[idx, 1],
                                     obs[idx, 5] - np.hypot(cmd[idx, 0], cmd[idx, 1])], 1)), 1)
        bad["obs_cmd"] += int((e > TOL).sum())
        for s in np.unique(seg[idx]):
            lab = labels[s]
            r = rows.get(lab)
            if r is None:
                continue
            hs = idx[seg[idx] == s]
            want = np.array([r["vx"], r["vy"], r["wz"]], dtype=np.float32)
            bad["cmd_row"] += int((np.abs(cmd[hs] - want) > TOL).any(axis=1).sum())
            if r["wz"] != 0.0:
                hs2 = hs[int(round(1.0 / dt)):]
                turn_steps += hs2.size
                bad["clock_running"] += int((~z["clock_running"][hs2].astype(bool)).sum())
                bad["clock_speed"] += int((z["clock_cmd_speed"][hs2] < FLOOR - TOL).sum())
                dph = np.abs(np.diff(np.unwrap(z["clock_phase"][hs2].astype(np.float64))))
                bad["clock_still"] += int((dph < 1e-6).sum())
    ok = (sum(v for k, v in bad.items() if k != "clock_still") == 0
          and (turn_steps == 0 or bad["clock_still"] <= 0.01 * turn_steps))
    return {"stamp": str(stamp), "route": route, "steps": steps, "turn_steps": turn_steps, "bad": bad, "pass": ok}


def check_world_rate(stamp: Path, w_min: float, w_max: float) -> dict:
    m = json.loads((stamp / "run_meta.json").read_text())
    route = (m.get("command_route"), m.get("command_adapter"), m.get("env_command_frame"))
    bad = {"obs_facing": 0, "obs_cmd": 0, "wz_outside": 0}
    steps = 0
    for f in sorted((stamp / "raw").glob("episode_*.npz")):
        z = np.load(f)
        seg = z["segment_id"].astype(int)
        idx = np.where((seg >= 0) & ~z["done"])[0]
        steps += idx.size
        obs, cmd = z["obs_cmd"], z["cmd_applied"]
        bad["obs_facing"] += int((np.abs(obs[idx, 1:3]) > 0).any(axis=1).sum())
        e = np.max(np.abs(np.stack([obs[idx, 0] - cmd[idx, 2], obs[idx, 3] - cmd[idx, 0], obs[idx, 4] - cmd[idx, 1],
                                     obs[idx, 5] - np.hypot(cmd[idx, 0], cmd[idx, 1])], 1)), 1)
        bad["obs_cmd"] += int((e > TOL).sum())
        w = np.abs(cmd[idx, 2])
        bad["wz_outside"] += int(((w > 0) & ((w < w_min - TOL) | (w > w_max + TOL))).sum())
    clipped = None
    try:
        agg = json.loads((stamp / "scenario_metrics.json").read_text())["aggregate"]
        clipped = {k: (v.get("clipped_frac") if isinstance(v, dict) else v)
                   for k, v in (agg.get("adapter_by_hold") or {}).items()}
    except (OSError, ValueError, KeyError):
        pass
    ok = route[1] == "world_rate" and route[2] == "body" and sum(bad.values()) == 0
    return {"stamp": str(stamp), "route": route, "steps": steps, "bad": bad, "clipped_by_hold": clipped, "pass": ok}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--term", action="append", default=[])
    ap.add_argument("--world-rate", action="append", default=[])
    ap.add_argument("--chain", action="append", default=[])
    ap.add_argument("--json", default=str(C / "step5_smoke.json"))
    a = ap.parse_args()
    env = json.loads((C / "envelope_freeze.json").read_text())["envelope"]
    res = {"term": [check_term(Path(s)) for s in a.term],
           "world_rate": [check_world_rate(Path(s), env["w_min"], env["w_max"]) for s in a.world_rate],
           "audits": {}}
    for ch in a.chain:
        p = C / "stage_reads" / f"{ch}.json"
        res["audits"][ch] = json.loads(p.read_text()).get("audit") if p.is_file() else "MISSING"
    res["pass"] = (all(x["pass"] for x in res["term"]) and all(x["pass"] for x in res["world_rate"])
                   and all(v == "PASS" for v in res["audits"].values()) and bool(a.chain))
    Path(a.json).write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=1))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
