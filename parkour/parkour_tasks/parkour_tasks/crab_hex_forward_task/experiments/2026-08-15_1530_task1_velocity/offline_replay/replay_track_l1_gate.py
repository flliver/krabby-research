"""C1 replay gate: linear |v_err| tracking penalty on saved traces (task1-velocity).
Gate: position-era ref (the only tracking-capable gait) pays < 10% of locomotion income;
every command-blind velocity-era family pays 3-25%; term is a pure penalty (unfarmable).
Usage: python replay_track_l1_gate.py"""
import sys
import numpy as np
from pathlib import Path

REPO = Path("/home/nickmagus/krabby/krabby-research")
BASE = REPO / "parkour/logs/rsl_rl/gait_eval/v1/flat_walk_forward/seed001"
W = -0.5
INCOME = 65.0  # per-min locomotion scale
FAMS = {
    "position-era ref (tracks)": REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-13_0035_mirror_symmetry/fromscratch_sym_20k/gait_eval_final/flat_walk_forward/seed001/2026-08-13_10-54-18",
    "B0 oscillator w=0": BASE / "2026-08-13_23-22-03",
    "spin ref (rescinded)": BASE / "2026-08-14_19-13-14",
    "sigma-fix screen": sorted(BASE.glob("2026-08-15_2*"))[-1] if sorted(BASE.glob("2026-08-15_2*")) else None,
}

def income_per_min(d):
    vals = []
    for f in sorted((d / "raw").glob("episode_*.npz")):
        z = np.load(f)
        err = np.linalg.norm(z["cmd_applied"][:, :2] - z["root_lin_vel_b"][:, :2], axis=1)
        st = z["steady_mask"].astype(bool)
        n = min(len(err), len(st))
        dt = float(z["dt"]) if "dt" in z else 0.02
        vals.append(err[:n][st[:n]].sum() * dt / (st[:n].sum() * dt / 60.0))
    return float(np.median(vals))

ok = True
for name, d in FAMS.items():
    if d is None or not (d / "raw").exists():
        print(f"[{name}] no traces, skipped"); continue
    inc = income_per_min(d)
    pen = abs(W) * inc
    frac = pen / INCOME
    print(f"[{name}] |err| income {inc:.1f}/min -> penalty {pen:.1f}/min = {frac*100:.1f}% of income")
    if "position-era" in name and frac > 0.10: ok = False
    if ("oscillator" in name or "spin ref" in name) and not (0.03 <= frac <= 0.25): ok = False
print("GATE:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
