"""C3 replay gate: stance-count band penalty (count outside {3,4} while commanded).
Task 1 s2.3's untried variant. Gate: ideal tripod ~0; healthy gaits low; fall-and-spin
(0 feet) and unison/shuffle families pay heavily. Usage: python replay_stance_band_gate.py"""
import sys
import numpy as np
from pathlib import Path

REPO = Path("/home/nickmagus/krabby/krabby-research")
BASE = REPO / "parkour/logs/rsl_rl/gait_eval/v1/flat_walk_forward/seed001"
W = -0.2
INCOME = 65.0
FAMS = {
    "position-era ref": REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-13_0035_mirror_symmetry/fromscratch_sym_20k/gait_eval_final/flat_walk_forward/seed001/2026-08-13_10-54-18",
    "C2 (tripod 0.573)": sorted(BASE.glob("2026-08-15_*"))[-1],
    "B0 oscillator": BASE / "2026-08-13_23-22-03",
    "fallen spinner (lottery s2)": BASE / "2026-08-15_03-34-14",
}

def frac_out_of_band(d):
    vals = []
    for f in sorted((d / "raw").glob("episode_*.npz")):
        z = np.load(f)
        contact = z["foot_force_norm"] > 1.0
        counts = contact.sum(axis=1)
        st = z["steady_mask"].astype(bool)
        n = min(len(counts), len(st))
        m = st[:n]
        if m.sum():
            vals.append(((counts[:n][m] < 3) | (counts[:n][m] > 4)).mean())
    return float(np.median(vals)) if vals else float("nan")

ok = True
res = {}
for name, d in FAMS.items():
    fr = frac_out_of_band(d)
    pen = abs(W) * fr * 60 / 0.02 * 0.02  # per-step frac * steps/min * dt = frac*60
    res[name] = pen
    print(f"[{name}] out-of-band {fr*100:.0f}% of steady steps -> penalty {pen:.1f}/min = {pen/INCOME*100:.1f}% of income")
# synthetic ideal tripod: perfect 3-3 alternation with brief 6-contact overlap (10%)
ideal_fr = 0.10
print(f"[ideal tripod, 10% overlap] penalty {abs(W)*ideal_fr*60:.1f}/min = {abs(W)*ideal_fr*60/INCOME*100:.1f}% of income")
if not (res["fallen spinner (lottery s2)"] > 3 * max(res["position-era ref"], 0.5)): ok = False
if res["position-era ref"] / INCOME > 0.10: ok = False
print("GATE:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
