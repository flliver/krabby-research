"""Offline replay gate for the π-recalibrated tripod band (gait-formation Phase 0).

Replays the ACTUAL repo function (`crab_hex_tripod_reward.tripod_swap_crossing_reward_step`)
with the RETUNED band (corr_tau 0.70, period band 0.30–1.40 — the 2.0 s gait cycle at
CAM_VEL_SCALE=π) over NEW-PLANT fixtures. Old-model fixtures are invalid (mass/geometry/
CAM_VEL_SCALE all changed 2026-08-20).

Gate criteria (adapted for a plant with no healthy gait yet):
  (a) a synthetic ideal tripod at the 2.0 s cam cycle must earn strongly (income alive
      at the target behavior — the "gradient-alive" check);
  (b) the smoke-2 wheelie rollout (the known degenerate basin) must earn ≈ 0;
  (c) the zero-action stand must earn ≈ 0;
  (d) sanity: the OLD band (0.10/0.60, corr_tau 0.20) must FAIL to pay the same ideal —
      proving the recalibration was necessary, not cosmetic.

Usage: python replay_tripod_band_gate.py
"""
import glob
import os
import sys

import numpy as np
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "parkour", "parkour_tasks", "parkour_tasks",
                                "crab_hex_forward_task", "mdp"))
from crab_hex_tripod_reward import (  # noqa: E402
    RESET_T_SINCE,
    S_T_SINCE,
    STATE_DIM,
    tripod_swap_crossing_reward_step,
)

FIXTURE_ROOT = os.path.join(_REPO, "parkour", "logs", "rsl_rl", "gait_eval",
                            "gait_formation_fixtures")
RUNS = {
    "wheelie": os.path.join(FIXTURE_ROOT, "flat_walk_forward_v2"),
    "stand": os.path.join(FIXTURE_ROOT, "flat_walk_slow_v2"),
}
WEIGHT = 0.30  # Phase-B mid arm weight
FORCE_THRESH = 1.0
TRIPOD_A = (0, 3, 4)
TRIPOD_B = (1, 2, 5)

NEW_BAND = dict(corr_tau=0.70, min_period=0.30, max_period=1.40)
OLD_BAND = dict(corr_tau=0.20, min_period=0.10, max_period=0.60)


def load_episodes(run_dir):
    npzs = sorted(glob.glob(os.path.join(run_dir, "**/raw/episode_*.npz"), recursive=True))
    eps = []
    for f in npzs:
        d = np.load(f)
        eps.append({
            "contact": d["foot_force_norm"] > FORCE_THRESH,
            "cmd_xy": d["cmd_applied"][:, :2].astype(np.float32),
            "steady": d["steady_mask"].astype(bool),
            "dt": float(d["dt"]),
        })
    return eps


def ideal_episode(T=1800, dt=0.02, period=2.0, overlap=0.15):
    """Ideal alternating tripod at the π-cam gait cycle (2.0 s, ~15% double support)."""
    contact = np.zeros((T, 6), bool)
    for t in range(T):
        ph = (t * dt) % period
        in_a = ph < period / 2 + overlap
        in_b = ph >= period / 2 or ph < overlap
        contact[t, list(TRIPOD_A)] = in_a
        contact[t, list(TRIPOD_B)] = in_b
    return {"contact": contact, "cmd_xy": np.full((T, 2), 0.45, np.float32),
            "steady": np.ones(T, bool), "dt": dt}


def replay(ep, **params):
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


def income_per_min(name, eps, band):
    mins = sum(ep["steady"].sum() * ep["dt"] for ep in eps) / 60.0
    inc = crossings = 0
    for ep in eps:
        i, c = replay(ep, **band)
        inc += i
        crossings += c
    rate = WEIGHT * inc / max(mins, 1e-9)
    print(f"{name:<22}{rate:>12.3f}{crossings / max(mins, 1e-9):>14.1f}")
    return rate


def main():
    print(f"{'trace (band)':<22}{'inc/min@0.30':>12}{'paid/min':>14}")
    rows = {}
    for name, run_dir in RUNS.items():
        eps = load_episodes(run_dir)
        if not eps:
            print(f"{name:<22}  NO NPZ FOUND under {run_dir}")
            return 2
        rows[name] = income_per_min(f"{name} (new)", eps, NEW_BAND)
    ideal = [ideal_episode()]
    rows["ideal"] = income_per_min("ideal-2s (new)", ideal, NEW_BAND)
    rows["ideal_old"] = income_per_min("ideal-2s (OLD band)", ideal, OLD_BAND)

    ok = (
        rows["ideal"] > 1.0                        # (a) income alive at the target
        and rows["wheelie"] < 0.2 * rows["ideal"]  # (b) degenerate dead
        and rows["stand"] < 0.2 * rows["ideal"]    # (c) static dead
        and rows["ideal_old"] < 0.2 * rows["ideal"]  # (d) old band provably broken
    )
    print(f"\nideal(new) {rows['ideal']:.3f} | wheelie {rows['wheelie']:.3f} | "
          f"stand {rows['stand']:.3f} | ideal(old band) {rows['ideal_old']:.3f}")
    print("GATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
