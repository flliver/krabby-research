```
=== crab-hex gait eval ===
scenario   : flat_walk_forward  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/2026-08-10_0058_tripod_stability/fromscratch_tripod_reward_w0.15/logs/rsl_rl/crab_hex_flat_walk/2026-08-10_20-08-53/model_19999.pt
episodes   : 10  unscored: 1
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.0  p25=0.0  p75=0.0
tippy_tap_fraction median=0.02105496453900709
slip_ratio         median=0.021999995719471407
schedule_completion_rate=0.5
terminations={'fall': 5, 'schedule_complete': 5}

by hold:
       low: tripod median=0.0 (n=9)
       mid: tripod median=0.0 (n=5)
      high: tripod median=0.0 (n=5)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
