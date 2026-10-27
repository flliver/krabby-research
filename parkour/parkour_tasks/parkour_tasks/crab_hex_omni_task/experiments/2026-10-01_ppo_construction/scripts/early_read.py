"""Side-speed bisection early read (CPU, read-only): track_ratio_mean and side progress around a
training iteration, from a run dir's tfevents.

Usage: early_read.py RUN_DIR [CENTER ...]   (default centers 500 900 1000 1500 2000 2500 2950)
The CHARTER's early read is track_ratio_mean averaged over its 800-1000: >= 0.58 forming FAST,
<= 0.48 forming SLOW (uncalibrated; it only holds C back, never stops an arm).
"""
import sys
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAGS = {
    "ratio": "Metrics/base_parkour/track_ratio_mean",
    "ratio_mv": "Metrics/base_parkour/track_ratio_move_mean",
    "prog": "Episode_Reward/reward_forward_progress_along_command",
    "trk": "Episode_Reward/track_lin_vel_xy_exp",
    "err": "Metrics/base_velocity/error_vel_xy",
    "eplen": "Train/mean_episode_length",
}


def main():
    d = Path(sys.argv[1])
    centers = [int(x) for x in sys.argv[2:]] or [500, 900, 1000, 1500, 2000, 2500, 2950]
    ea = EventAccumulator(str(d), size_guidance={"scalars": 0})
    ea.Reload()
    have = set(ea.Tags()["scalars"])
    ser = {}
    for k, t in TAGS.items():
        if t in have:
            s = ea.Scalars(t)
            ser[k] = (np.array([x.step for x in s]), np.array([x.value for x in s]))
    last = max(int(v[0].max()) for v in ser.values()) if ser else -1
    print(f"{d.name}: last it {last}")
    if "ratio" in ser:
        st, v = ser["ratio"]
        m = (st >= 800) & (st < 1000)
        if m.sum() >= 100:
            er = float(np.nanmean(v[m]))
            tag = "forming FAST" if er >= 0.58 else ("forming SLOW" if er <= 0.48 else "between")
            print(f"EARLY READ track_ratio_mean its 800-1000: {er:.3f} ({tag}, n {int(m.sum())})")
    for c in centers:
        if c - 50 > last:
            continue
        parts = []
        for k, (st, v) in ser.items():
            m = (st >= c - 50) & (st < c + 50)
            if m.any():
                parts.append(f"{k} {np.nanmean(v[m]):.3f}")
        print(f"  it {c:5d}: " + "  ".join(parts))


if __name__ == "__main__":
    main()
