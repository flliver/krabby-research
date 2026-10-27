"""Extract loss/metric scalars for the side-bisect arms into a pickle cache (CPU, read-only)."""
import glob, pickle, sys
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
R = "/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/"
RUNS = {  # label: (run-dir prefix, class, directions)
 "O": ("2026-09-29_00-40-18", "S", "+y/4img"), "C": ("2026-09-29_01-52-56", "F", "+y/fa"),
 "R": ("2026-09-29_08-28-30", "S", "+y/4img"), "C_mo": ("2026-09-29_09-39-22", "I", "+y/fa"),
 "C_band": ("2026-09-29_10-52-23", "S", "+y/4img"), "C_rsi": ("2026-09-29_12-09-37", "S", "+y/4img"),
 "C_s2": ("2026-09-29_13-23-08", "F", "+y/fa"), "C_mo_s2": ("2026-09-29_14-36-09", "F", "+y/fa"),
 "C_mo_s1": ("2026-09-29_18-09-45", "F", "+y/fa"), "LRfa": ("2026-09-29_19-18-51", "S", "±y/fa"),
 "LR4": ("2026-09-29_20-33-42", "S", "±y/4img"), "LR4_dwell": ("2026-09-29_22-15-30", "D", "±y/4img"),
 "C_lr": ("2026-09-30_00-00-38", "S", "±y/fa"), "C_lr_10": ("2026-09-30_01-15-33", "S", "±y/fa"),
 "C_lr_rsi": ("2026-09-30_09-05-43", "S", "±y/fa"), "C_lr_10_rsi": ("2026-09-30_10-35-34", "S/D", "±y/fa"),
 "C_lr_rsi33": ("2026-09-30_12-27-45", "?", "±y/fa"),
}
TAGS = ["Loss/symmetry", "Loss/surrogate", "Loss/value_function", "Loss/entropy", "Loss/priv_reg",
        "Loss/priv_reg_coef", "Loss/learning_rate", "Loss/estimator", "Loss/hist_latent", "Policy/mean_noise_std",
        "Train/mean_reward", "Metrics/base_parkour/track_ratio_mean", "Episode_Reward/track_lin_vel_xy_exp"]
out = {}
for lab, (pre, cls, dirs) in RUNS.items():
    ds = glob.glob(R + pre + "*")
    if not ds:
        print("missing", lab); continue
    ea = EventAccumulator(ds[0], size_guidance={"scalars": 0}); ea.Reload()
    have = ea.Tags()["scalars"]
    rec = {"dir": ds[0], "cls": cls, "dirs": dirs}
    for t in TAGS:
        if t in have:
            ev = ea.Scalars(t)
            rec[t] = np.array([[e.step, e.value] for e in ev])
    out[lab] = rec
    print(lab, ds[0].split("/")[-1], {t.split("/")[-1]: len(rec[t]) for t in TAGS if t in rec}.get("symmetry"))
pickle.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "tb_cache.pkl", "wb"))
