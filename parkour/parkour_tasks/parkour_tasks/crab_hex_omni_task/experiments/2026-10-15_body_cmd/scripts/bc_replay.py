#!/usr/bin/env python3
"""Step 4 (CPU): offline replay of the body-mode yaw-term set on S1's saved traces (PLAN step 4, D4).

Uses the SHIPPED pure income functions (``mdp/crab_hex_omni_yaw_window.py``: ``yaw_window_income``,
``yaw_tent_income``), loaded by path, over full trailing windows of ``window_steps(2.0, dt)`` samples, at the realised
weights of the ``_b`` presets (window term 0.2, tent 2.0; the straight-gated ang-vel term pays 0 on turning rows).

Traces (REFS_S1_s18 ``suite__omni`` es1 + es2, per-step ``root_ang_vel_b[:, 2]``, signed toward each hold's own
turn so every turn trace reads positive):
  healthy turn     turn_l90 / turn_r90 first 6 s (relay saturated)            under cmd = ceiling
  healthy move     move_turn / move_turn_r first 6 s                          under cmd = cap80 (0.8 cap(0.22))
  healthy synth    translation wobble (mean removed) + c                      under cmd = c (the step-1a tracker)
Degenerates, each read as a share of the healthy income at the same command (PLAN: "within the step-1a margins" --
<= 0.40 at w_min, <= 0.10 at the ceiling):
  over-spin        healthy turn traces (~ceiling) under cmd = w_min           vs healthy synth at w_min
  stand-on-turn    stand traces under cmd = ceiling / w_min                    vs healthy turn / healthy synth
  step-in-place    translation wobble (mean removed) under cmd = w_min        vs healthy synth at w_min
Reported only: a 0.01 rad/s drift on straight rows (wobble + 0.01 under cmd 0: window + ang-vel terms).

    bc_replay.py --envelope envelope_freeze.json [--json OUT] [--md OUT]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
R = Path("/home/nickmagus/krabby/krabby-research")
YW_PY = R / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/mdp/crab_hex_omni_yaw_window.py"
spec = importlib.util.spec_from_file_location("_bc_yw", YW_PY)
YW = importlib.util.module_from_spec(spec)
spec.loader.exec_module(YW)
REFS = R / "parkour/outputs/omni_plan/phase2/lr/REFS_S1_s18/suite__omni"
STAMPS = (REFS / "seed001/2026-10-09_03-20-24", REFS / "seed002/2026-10-09_03-27-23")
W_TRACK, W_TENT, W_ANG, ANG_STD = 0.2, 2.0, 1.0, 0.5
TRANSLATE = ("fwd", "lat_l", "lat_r", "back", "diag")


def traces():
    out = {"turn": [], "move": [], "stand": [], "translate": []}
    dt = None
    for st in STAMPS:
        m = json.loads((st / "run_meta.json").read_text())
        dt = float(m["dt"])
        n6 = int(round(6.0 / dt))
        for row in m["env_index_map"]:
            lab = row["label"]
            z = np.load(st / "raw" / f"episode_{row['env_index']:02d}.npz")
            h = z["segment_id"] >= 1
            if z["done"][h].any():
                continue
            wz = z["root_ang_vel_b"][h, 2].astype(np.float64)
            if lab in ("turn_l90", "turn_r90"):
                out["turn"].append((1.0 if lab == "turn_l90" else -1.0) * wz[:n6])
            elif lab in ("move_turn", "move_turn_r"):
                out["move"].append((1.0 if lab == "move_turn" else -1.0) * wz[:n6])
            elif lab == "stand":
                out["stand"].append(wz[100:])
            elif lab in TRANSLATE:
                out["translate"].append(wz[100:])
    return out, dt


def income(wz_traces, c: float, dt: float, w_max: float, n: int) -> float:
    """Mean per-step yaw-term income of the traces under a constant command c (full windows only)."""
    vals = []
    for wz in wz_traces:
        x = torch.as_tensor(wz, dtype=torch.float64)
        cs = torch.cat([torch.zeros(1, dtype=torch.float64), torch.cumsum(x, 0)]) * dt
        s_ach = cs[n:] - cs[:-n]
        s_cmd = torch.full_like(s_ach, c * n * dt)
        win = YW.yaw_window_income(s_ach, s_cmd, YW.YAW_WINDOW_SIGMA)
        tent = YW.yaw_tent_income(s_ach, s_cmd, n * dt, w_max)
        v = W_TRACK * win + W_TENT * tent
        if c == 0.0:
            v = v + W_ANG * torch.exp(-torch.square(x[n - 1:]) / ANG_STD ** 2)
        vals.append(v)
    return float(torch.cat(vals).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envelope", default=str(HERE.parent / "envelope_freeze.json"))
    ap.add_argument("--json", default=str(HERE.parent / "step4_replay.json"))
    ap.add_argument("--md", default=str(HERE.parent / "step4_replay.md"))
    a = ap.parse_args()
    env = json.loads(Path(a.envelope).read_text())["envelope"]
    w_min, ceil_, w_max = float(env["w_min"]), float(env["ceiling"]), float(env["w_max"])
    cap = env["cap"]
    cap022 = float(np.interp(0.22, [p[0] for p in cap], [p[1] for p in cap]))
    cap80 = 0.8 * cap022
    tr, dt = traces()
    n = YW.window_steps(YW.YAW_WINDOW_S, dt)
    wob = [w - w.mean() for w in tr["translate"]]
    H = {
        "healthy_turn@ceiling": income(tr["turn"], ceil_, dt, w_max, n),
        "healthy_move@cap80": income(tr["move"], cap80, dt, w_max, n),
        "healthy_synth@w_min": income([w + w_min for w in wob], w_min, dt, w_max, n),
        "healthy_synth@ceiling": income([w + ceil_ for w in wob], ceil_, dt, w_max, n),
    }
    D = {
        "over_spin@w_min": (income(tr["turn"], w_min, dt, w_max, n), "healthy_synth@w_min", 0.40),
        "stand_on_turn@ceiling": (income(tr["stand"], ceil_, dt, w_max, n), "healthy_turn@ceiling", 0.10),
        "stand_on_turn@w_min": (income(tr["stand"], w_min, dt, w_max, n), "healthy_synth@w_min", 0.40),
        "step_in_place@w_min": (income(wob, w_min, dt, w_max, n), "healthy_synth@w_min", 0.40),
        "step_in_place@ceiling": (income(wob, ceil_, dt, w_max, n), "healthy_turn@ceiling", 0.10),
    }
    deg = {k: {"income": v, "vs": ref, "share": v / H[ref], "margin": mg, "pass": v / H[ref] <= mg}
           for k, (v, ref, mg) in D.items()}
    drift = {"straight_wobble": income(wob, 0.0, dt, w_max, n),
             "straight_drift_0p01": income([w + 0.01 for w in wob], 0.0, dt, w_max, n)}
    turn_rate = float(np.mean([w.mean() for w in tr["turn"]]))
    move_rate = float(np.mean([w.mean() for w in tr["move"]]))
    res = {"envelope": env, "cap80": cap80, "dt": dt, "window_steps": n, "weights": [W_TRACK, W_TENT, W_ANG],
           "n_traces": {k: len(v) for k, v in tr.items()}, "healthy": H, "degenerate": deg, "drift_reported": drift,
           "trace_mean_rate": {"turn": turn_rate, "move": move_rate},
           "pass": all(d["pass"] for d in deg.values())}
    Path(a.json).write_text(json.dumps(res, indent=2) + "\n")
    L = ["# Step 4 — offline replay of the body yaw-term set", "",
         f"Shipped pure functions (`crab_hex_omni_yaw_window.py`), window {n} steps, weights window {W_TRACK} / tent "
         f"{W_TENT}; envelope {env}; cap80 = {cap80:.4f}. Traces: {res['n_traces']}; S1 turn traces' mean rate "
         f"{turn_rate:.4f}, move traces' {move_rate:.4f} rad/s.", "", f"**Verdict: {'PASS' if res['pass'] else 'FAIL'}**", "",
         "| healthy | income / step |", "|---|---|"]
    L += [f"| {k} | {v:.4f} |" for k, v in H.items()]
    L += ["", "| degenerate | income / step | vs | share | margin | pass |", "|---|---|---|---|---|---|"]
    L += [f"| {k} | {d['income']:.4f} | {d['vs']} | {d['share']:.3f} | {d['margin']} | {d['pass']} |"
          for k, d in deg.items()]
    L += ["", f"Reported only: straight-row yaw income of S1's wobble {drift['straight_wobble']:.4f} / step; with a "
          f"0.01 rad/s drift {drift['straight_drift_0p01']:.4f} / step."]
    Path(a.md).write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
