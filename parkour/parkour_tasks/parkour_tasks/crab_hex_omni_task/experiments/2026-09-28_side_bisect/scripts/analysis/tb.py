import glob, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
R = "/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/"
TAG = {"O":"2026-09-29_00-40-18","C":"2026-09-29_01-52-56","R":"2026-09-29_08-28-30","C_mo":"2026-09-29_09-39-22",
       "C_band":"2026-09-29_10-52-23","C_rsi":"2026-09-29_12-09-37","C_s2":"2026-09-29_13-23-08","C_mo_s2":"2026-09-29_14-36-09",
       "C_mo_s1":"2026-09-29_18-09-45","LRfa":"2026-09-29_19-18-51","LR4":"2026-09-29_20-33-42","LR4_dwell":"2026-09-29_22-15-30",
       "C_lr":"2026-09-30_00-00-38","C_lr_10":"2026-09-30_01-15-33"}
if sys.argv[1] == "tags":
    d = glob.glob(R + TAG["O"] + "*")[0]
    ea = EventAccumulator(d, size_guidance={"scalars": 0}); ea.Reload()
    for t in ea.Tags()["scalars"]: print(t)
    sys.exit()
want = sys.argv[1].split(",")
its = [200, 500, 1000, 2000, 2990]
for arm, ts in TAG.items():
    d = glob.glob(R + ts + "*")[0]
    ea = EventAccumulator(d, size_guidance={"scalars": 0}); ea.Reload()
    row = []
    for w in want:
        if w not in ea.Tags()["scalars"]:
            row.append(f"{w}: n/a"); continue
        ev = ea.Scalars(w); s = np.array([e.step for e in ev]); v = np.array([e.value for e in ev])
        vals = [np.mean(v[(s >= i - 50) & (s <= i + 50)]) if ((s >= i - 50) & (s <= i + 50)).any() else np.nan for i in its]
        row.append(w.split("/")[-1] + " " + " ".join(f"{x:.3g}" for x in vals))
    print(f"{arm:9s} | " + " | ".join(row))
