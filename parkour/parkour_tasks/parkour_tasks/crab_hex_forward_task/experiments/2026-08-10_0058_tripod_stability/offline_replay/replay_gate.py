"""Offline replay gate for candidate tripod reward terms (mandatory before any training screen).

Replays the ACTUAL repo reward function (`crab_hex_tripod_reward.tripod_swap_crossing_reward_step`)
over saved gait-eval npz traces and reports weighted income per minute of steady walking.

Gate criteria (see RESULTS.md "v4 re-eval" post-mortem for why this exists):
  (a) the healthy refs (H2000/H3000/H5000) must earn meaningfully;
  (b) every known degenerate basin (v1 lunge, v2 tip-rock, v3 skate, v3b drag) must earn ~0;
  (c) a synthetic ideal tripod trace must earn well above healthy (the optimum sits at the
      target behavior, giving training a slope from the baseline's shallow taps toward it).

v1-v4 all skipped check (a) and all four failed screens (~1h GPU each) would have been caught
here for free. Usage:  python replay_gate.py
"""
import glob
import os
import sys

import numpy as np
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "parkour", "parkour_tasks", "parkour_tasks",
                                "crab_hex_forward_task", "mdp"))
from crab_hex_tripod_reward import (  # noqa: E402
    RESET_T_SINCE,
    S_T_SINCE,
    STATE_DIM,
    tripod_swap_crossing_reward_step,
)

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS = {
    "H2000": f"{BASE}/healthy_refs/gait_eval_h2000",
    "H3000": f"{BASE}/healthy_refs/gait_eval_h3000",
    "H5000": f"{BASE}/fromscratch_tripod_v4_swap_short/gait_eval_h5000",
    "v1-lunge": f"{BASE}/fromscratch_tripod_reward_w0.15/gait_eval",
    "v2-tiprock": f"{BASE}/fromscratch_tripod_v2_w0.15/gait_eval_midtrain",
    "v3-skate": f"{BASE}/fromscratch_tripod_v3_w0.15_short/gait_eval_3000",
    "v3b-drag": f"{BASE}/fromscratch_tripod_v3b_slide_short/gait_eval_3000",
    "v4-falls": f"{BASE}/fromscratch_tripod_v4_swap_short/gait_eval_5000",
}
WEIGHT = 0.15
FORCE_THRESH = 1.0  # N; same raw-contact threshold score_gait's engagement check uses
TRIPOD_A = (0, 3, 4)
TRIPOD_B = (1, 2, 5)


def load_episodes(run_dir):
    npzs = sorted(glob.glob(os.path.join(run_dir, "**/raw/episode_*.npz"), recursive=True))
    eps = []
    for f in npzs:
        d = np.load(f)
        eps.append({
            "contact": d["foot_force_norm"] > FORCE_THRESH,
            "cmd_xy": d["cmd_applied"][:, :2].astype(np.float32),
            "steady": d["steady_mask"],
            "dt": float(d["dt"]),
        })
    return eps


def ideal_episode(T=2925, dt=0.02, period=0.30, overlap=0.03):
    contact = np.zeros((T, 6), bool)
    for t in range(T):
        ph = (t * dt) % period
        in_a = ph < period / 2 + overlap
        in_b = ph >= period / 2 or ph < overlap
        contact[t, list(TRIPOD_A)] = in_a
        contact[t, list(TRIPOD_B)] = in_b
    return {"contact": contact, "cmd_xy": np.full((T, 2), 0.5, np.float32),
            "steady": np.ones(T, bool), "dt": dt}


def replay(ep, **params):
    """Income and paid-crossing count over steady steps (unweighted)."""
    state = torch.zeros(1, STATE_DIM)
    state[:, S_T_SINCE] = RESET_T_SINCE
    cmd = torch.from_numpy(ep["cmd_xy"])
    total = 0.0
    paid = 0
    for t in range(ep["contact"].shape[0]):
        contact = torch.from_numpy(ep["contact"][t:t + 1])
        r, state = tripod_swap_crossing_reward_step(contact, cmd[t:t + 1], state, ep["dt"], **params)
        if ep["steady"][t] and float(r) > 0.0:
            total += float(r)
            paid += 1
    return total, paid


def main():
    print(f"{'run':<14}{'inc/min@0.15':>14}{'paid cross/min':>16}")
    rows = {}
    for name, run_dir in list(RUNS.items()) + [("IDEAL-tripod", None)]:
        eps = [ideal_episode()] if run_dir is None else load_episodes(run_dir)
        if not eps:
            print(f"{name:<14}  NO NPZ FOUND: {run_dir}")
            continue
        mins = sum(ep["steady"].sum() * ep["dt"] for ep in eps) / 60.0
        inc = crossings = 0
        for ep in eps:
            i, c = replay(ep)
            inc += i
            crossings += c
        rows[name] = WEIGHT * inc / mins
        print(f"{name:<14}{rows[name]:>14.3f}{crossings / mins:>16.1f}")

    healthy = min(rows[k] for k in ("H2000", "H3000", "H5000"))
    degen = max(rows[k] for k in ("v1-lunge", "v2-tiprock", "v3-skate", "v3b-drag"))
    print(f"\nhealthy-min {healthy:.3f} vs degenerate-max {degen:.3f} "
          f"(ratio {healthy / max(degen, 1e-9):.0f}:1); ideal {rows['IDEAL-tripod']:.3f}")
    ok = healthy > 1.0 and degen < 0.2 * healthy and rows["IDEAL-tripod"] > 5 * healthy
    print("GATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
