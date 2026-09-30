"""Independent tfevents recompute (no reuse of tb/*.pkl)."""
import glob, json, sys
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
R = "/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/"
RUNS = {"O":"2026-09-29_00-40-18","C":"2026-09-29_01-52-56","R":"2026-09-29_08-28-30","C_mo":"2026-09-29_09-39-22",
 "C_band":"2026-09-29_10-52-23","C_rsi":"2026-09-29_12-09-37","C_s2":"2026-09-29_13-23-08","C_mo_s2":"2026-09-29_14-36-09",
 "C_mo_s1":"2026-09-29_18-09-45","LRfa":"2026-09-29_19-18-51","LR4":"2026-09-29_20-33-42","LR4_dwell":"2026-09-29_22-15-30",
 "C_lr":"2026-09-30_00-00-38","C_lr_10":"2026-09-30_01-15-33"}
CLS = {"O":"S","R":"S","C_band":"S","C_rsi":"S","LRfa":"S","LR4":"S","LR4_dwell":"D","C_lr":"S","C_lr_10":"S",
       "C":"F","C_s2":"F","C_mo_s2":"F","C_mo_s1":"F","C_mo":"I"}
def load(arm):
    d = glob.glob(R + RUNS[arm] + "*")[0]
    ea = EventAccumulator(d, size_guidance={"scalars": 0}); ea.Reload()
    out = {}
    for t in ea.Tags()["scalars"]:
        ev = ea.Scalars(t); out[t] = (np.array([e.step for e in ev]), np.array([e.value for e in ev]))
    return out
def w(D, t, a, b):
    if t not in D: return np.nan
    s, v = D[t]; m = (s >= a) & (s < b)
    return float(np.mean(v[m])) if m.any() else np.nan
data = {a: load(a) for a in RUNS}
tags = sorted(data["O"].keys())
if "tags" in sys.argv:
    print("\n".join(tags))
TR = "Metrics/base_parkour/track_ratio_mean"; TRM = "Metrics/base_parkour/track_ratio_move_mean"
RW = [t for t in tags if "mean_reward" in t]; print("reward tags:", RW)
TIB = [t for t in tags if "tibia" in t]; print("tibia tags:", TIB)
STD = [t for t in tags if "noise_std" in t]; print("std tags:", STD)
print("\n%-9s c  early(800-1000) trm800-1000 final(2500-3000) rew2500-3000 rew2900-3000 tib(2500-3000) tib(800-1000) std@3000 std>=1.99first" % "arm")
for a in RUNS:
    D = data[a]
    std = D[STD[0]][1] if STD else None; s_steps = D[STD[0]][0] if STD else None
    first = int(s_steps[np.argmax(std >= 1.99)]) if STD and (std >= 1.99).any() else -1
    tib = TIB[0] if TIB else None
    print("%-9s %s  %.3f  %.3f  %.3f  %.2f  %.2f  %.4f  %.4f  %.3f  %d" % (a, CLS[a], w(D, TR, 800, 1000), w(D, TRM, 800, 1000), w(D, TR, 2500, 3000),
          w(D, RW[0], 2500, 3000) if RW else np.nan, w(D, RW[0], 2900, 3000) if RW else np.nan,
          w(D, tib, 2500, 3000) if tib else np.nan, w(D, tib, 800, 1000) if tib else np.nan, w(D, STD[0], 2950, 3000) if STD else np.nan, first))
# separation of track_ratio_mean: first it after which min(fast 50-it roll) > max(slow 50-it roll) for all later its
def roll(D, t, n=50):
    s, v = D[t]; full = np.full(int(s.max()) + 1, np.nan); full[s.astype(int)] = v
    full = np.where(np.isnan(full), np.nanmean(full), full)
    c = np.convolve(full, np.ones(n) / n, mode="valid"); return np.concatenate([np.full(n - 1, np.nan), c])
F = [a for a in RUNS if CLS[a] == "F"]; S = [a for a in RUNS if CLS[a] in "SD"]
rf = np.array([roll(data[a], TR) for a in F]); rs = np.array([roll(data[a], TR) for a in S])
L = min(rf.shape[1], rs.shape[1]); rf = rf[:, :L]; rs = rs[:, :L]
sep = rf.min(0) > rs.max(0)
last_bad = np.where(~sep[100:])[0]
print("track_ratio_mean 50-it roll: last it where min(FAST) <= max(SLOW):", int(last_bad[-1]) + 100 if len(last_bad) else None)
rf2 = np.array([roll(data[a], TR) for a in F + ["C_mo"]])[:, :L]
sep2 = rf2.min(0) > rs.max(0); lb2 = np.where(~sep2[100:])[0]
print("  incl C_mo: last it where min(FAST+INT) <= max(SLOW):", int(lb2[-1]) + 100 if len(lb2) else None)
# same-until-300 and plateau timing
print("\n%-9s c  tr@100 tr@200 tr@300 tr@400 tr@600 tr@1000 tr@1500 tr@2000 tr@3000  it95%%final  slope500->1500" % "arm")
for a in RUNS:
    r = roll(data[a], TR); fin = w(data[a], TR, 2500, 3000)
    p95 = int(np.argmax(r >= 0.95 * fin))
    vals = [r[i - 1] for i in (100, 200, 300, 400, 600, 1000, 1500, 2000, 3000) if i - 1 < len(r)]
    print("%-9s %s  " % (a, CLS[a]) + " ".join("%.3f" % x for x in vals) + "  %5d  %+.3f" % (p95, r[1499] - r[499]))
