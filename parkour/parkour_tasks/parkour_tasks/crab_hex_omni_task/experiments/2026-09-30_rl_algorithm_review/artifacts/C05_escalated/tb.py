import glob, sys, math
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
for run in ["/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-04_13-18-07",
            "/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-09-28_14-37-59_seed2_X02_c6y_A15pB",
            "/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-09-30_10-35-34_seed3_X17_s1_A15pB"]:
    ea = EventAccumulator(run, size_guidance={'scalars': 0}); ea.Reload()
    tags = ea.Tags()['scalars']
    print("RUN", run.split('/')[-1], "ntags", len(tags))
    for tag in ["Policy/mean_noise_std", "Loss/entropy"]:
        if tag not in tags: print("  missing", tag); continue
        ev = ea.Scalars(tag); v=[e.value for e in ev]; s=[e.step for e in ev]
        first_ge = next((s[i] for i in range(len(v)) if v[i] >= 1.999), None) if 'std' in tag else None
        print(f"  {tag}: n={len(v)} step[{s[0]}..{s[-1]}] first={v[0]:.5f} last={v[-1]:.5f} max={max(v):.5f} min={min(v):.5f} first>=1.999@{first_ge} n_gt2={sum(1 for x in v if x>2.0)} n_last500_eq_last={sum(1 for x in v[-500:] if abs(x-v[-1])<1e-6)}")
print("18*H(N(0,2)) =", 18*(0.5*math.log(2*math.pi*math.e)+math.log(2)))
