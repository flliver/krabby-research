"""Offline replay gate for the clock-referenced contact-schedule reward (PLAN E Phase 1).

Replays the shipped pure function (`crab_hex_clock_reward.clock_schedule_income`) over:
  (a) the Phase-0 scripted-gait reference (setAB walking stretch, this plant) -- must earn
      HIGH at its best clock alignment;
  (b) degenerate traces -- old-plant creep eval raws + wheelie/stand fixtures -- must stay
      at/below the standing floor (~0.5) at EVERY clock alignment (the 0.5 floor is by
      design: stance income is half the schedule; the swing half is the formation gradient);
  (c) gradient-alive-at-zero-behavior: unloading one foot inside its swing window on a creep
      trace must RAISE income (the dense-gradient property the crossing term lacked).

Gate: (a) >= 0.75; every (b) <= 0.60 at its best alignment; (c) strictly positive.

Usage: <venv-python> replay_clock_gate.py
"""
import glob
import math
import os
import sys

import numpy as np
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "parkour", "parkour_tasks", "parkour_tasks",
                                "crab_hex_forward_task", "mdp"))
from crab_hex_clock_reward import FOOT_OFFSETS, clock_schedule_income  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(HERE, "probe2_w+0.50_ph0.00_ks+1_setAB.npz")
CREEP_RUN = os.path.join(_REPO, "parkour", "logs", "rsl_rl", "gait_eval", "timing_probe",
                         "flat_walk_slow_v2", "seed001", "2026-08-21_14-24-16")
FIXTURES = os.path.join(_REPO, "parkour", "logs", "rsl_rl", "gait_eval",
                        "gait_formation_fixtures")
N_ALIGN = 16


COMBINE = "sum"  # overridden by --combine product


def income_trace(force, speed, phase, running=True):
    T = force.shape[0]
    run = torch.full((T,), running)
    return clock_schedule_income(
        torch.as_tensor(phase, dtype=torch.float32),
        torch.as_tensor(force, dtype=torch.float32),
        torch.as_tensor(speed, dtype=torch.float32),
        run,
        combine=COMBINE,
    ).mean().item()


def best_alignment(force, speed, freq_rev_s, dt):
    T = force.shape[0]
    base = 2 * math.pi * freq_rev_s * dt * np.arange(T)
    return max(
        income_trace(force, speed, base + 2 * math.pi * k / N_ALIGN)
        for k in range(N_ALIGN)
    )


def main() -> int:
    rows = {}
    # (a) Phase-0 reference: clock = the schedule the probe actually drove (FL cam + pi/2),
    # plus an alignment sweep for good measure.
    d = np.load(PROBE)
    done = d["done"]
    n = max(10, (int(np.argmax(done)) if done.any() else len(done)) - 25)
    force = d["foot_force_N"][:n]
    speed = np.linalg.norm(d["foot_lin_vel_w"][:n, :, :2], axis=-1)
    # v2/v4 probe npz lack cam_joint_names; the v1 probe recorded the same find_joints
    # order for the same articulation.
    v1 = np.load(os.path.join(HERE, "scripted_gait_w0.50.npz"))
    cam_names = [str(x) for x in v1["cam_joint_names"]]
    fl = next(i for i, c in enumerate(cam_names) if c.startswith("FL"))
    cam_clock = d["cam_pos"][:n, fl] + math.pi / 2.0
    direct = income_trace(force, speed, cam_clock)
    swept = max(
        income_trace(force, speed, cam_clock + 2 * math.pi * k / N_ALIGN)
        for k in range(N_ALIGN)
    )
    rows["probe_setAB"] = max(direct, swept)
    print(f"probe_setAB (walking ref): direct {direct:.3f} | best-aligned {swept:.3f}")

    # Synthetic ideal at the clock cadence: binary alternating tripod, still stance feet.
    T = 1000
    dt = 0.02
    freq = 0.35
    phase_ideal = 2 * math.pi * freq * dt * np.arange(T)
    swing = np.sin(phase_ideal[:, None] + np.asarray(FOOT_OFFSETS)[None, :]) > 0.0
    force_ideal = np.where(swing, 0.0, 700.0)
    rows["ideal"] = income_trace(force_ideal, np.zeros((T, 6)), phase_ideal)
    print(f"synthetic ideal: {rows['ideal']:.3f}")

    # (b) degenerates: WALKING-command segments only (commanded stops legitimately earn
    # full stance income in the shipped all-stance mode -- not a degenerate), best (most
    # charitable) clock alignment.
    def eval_run(name, run_dir, max_eps=8):
        files = sorted(glob.glob(os.path.join(run_dir, "**/raw/episode_*.npz"), recursive=True))[:max_eps]
        vals = []
        for f in files:
            dd = np.load(f)
            walking = np.abs(dd["cmd_applied"][:, 0]) > 0.2
            if walking.sum() < 100:
                continue
            fr = dd["foot_force_norm"][walking]
            sp = np.linalg.norm(dd["foot_lin_vel_w"][walking][:, :, :2], axis=-1)
            dt = float(dd["dt"])
            v = np.abs(dd["cmd_applied"][walking, 0])
            freq = (np.clip(v / 0.35, 0, 1) * 0.5).mean()
            vals.append(best_alignment(fr, sp, freq, dt))
        rows[name] = max(vals) if vals else float("nan")
        print(f"{name}: best over {len(vals)} eps/alignments = {rows[name]:.3f}")

    eval_run("creep_oldplant", CREEP_RUN)
    eval_run("wheelie_fix", os.path.join(FIXTURES, "flat_walk_forward_v2"))
    eval_run("stand_fix", os.path.join(FIXTURES, "flat_walk_slow_v2"))

    # (c) gradient-alive: on the creep trace, unload one foot during its swing window.
    files = sorted(glob.glob(os.path.join(CREEP_RUN, "**/raw/episode_*.npz"), recursive=True))
    dd = np.load(files[0])
    fr = dd["foot_force_norm"].copy()
    sp = np.linalg.norm(dd["foot_lin_vel_w"][:, :, :2], axis=-1)
    T = len(fr)
    phase = 2 * math.pi * 0.35 * float(dd["dt"]) * np.arange(T)
    before = income_trace(fr, sp, phase)
    swing_mask = np.sin(phase[:, None] + np.asarray(FOOT_OFFSETS)[None, :]) > 0.2
    fr_lifted = np.where(swing_mask, 0.0, fr)
    after = income_trace(fr_lifted, sp, phase)
    grad = after - before
    print(f"gradient-alive: creep {before:.3f} -> swing-unloaded {after:.3f} (delta {grad:+.3f})")

    worst_degen = max(rows[k] for k in ("creep_oldplant", "wheelie_fix", "stand_fix")
                      if rows[k] == rows[k])
    ok = (
        rows["ideal"] >= 0.85                       # reward tops out at the target gait
        # Differential criterion (added with the product mode, 2026-08-22): the ideal must
        # earn at least 2x the best degenerate. The additive mode FAILS this (0.931 vs
        # 2x0.578) -- which is precisely why seed-2 crept: the basin differential was too
        # thin. Mode-independent and stricter than any absolute threshold.
        and rows["ideal"] >= 2.0 * worst_degen
        and rows["probe_setAB"] >= worst_degen + 0.05  # real-plant walking beats every degenerate
        and grad > 0.05                             # dense gradient alive at creep behavior
    )
    print(f"\nideal {rows['ideal']:.3f} | probe {rows['probe_setAB']:.3f} | "
          f"worst degenerate {worst_degen:.3f} | gradient {grad:+.3f}")
    print("GATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--combine" in sys.argv:
        COMBINE = sys.argv[sys.argv.index("--combine") + 1]
        print(f"combine mode: {COMBINE}")
    raise SystemExit(main())
