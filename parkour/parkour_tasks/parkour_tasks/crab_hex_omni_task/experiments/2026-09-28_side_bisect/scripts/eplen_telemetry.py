"""Per-rung obstacle telemetry for the episode-length ladder (review F1/F2; CPU, read-only).
Usage: eplen_telemetry.py X17 X24 ...   -> means over its 2900-3000 of the exposure / failure scalars."""
import glob
import sys

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

KEYS = ["r_max_mean", "reach_obst_frac", "reach_edge_frac", "block_steps_mean", "crab_failure_flat",
        "crab_failure_obst", "crab_failure", "stand_frac_actual"]
print("xid   " + "  ".join(f"{k:>17s}" for k in KEYS))
for x in sys.argv[1:]:
    d = sorted(glob.glob(f"logs/rsl_rl/crab_hex_omni_flat_walk/*_{x}_s1_A15pB"))[-1]
    ea = EventAccumulator(d, size_guidance={"scalars": 0}); ea.Reload()
    tags = {t.split("/")[-1]: t for t in ea.Tags()["scalars"]}
    row = []
    for k in KEYS:
        if k not in tags:
            row.append(f"{'-':>17s}"); continue
        s = np.array([(e.step, e.value) for e in ea.Scalars(tags[k])])
        m = (s[:, 0] >= 2900) & (s[:, 0] < 3000)
        row.append(f"{np.nanmean(s[m, 1]):17.3f}" if m.any() else f"{'-':>17s}")
    print(f"{x:5s} " + "  ".join(row))
