```
=== crab-hex gait eval ===
scenario   : flat_walk_forward  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/2026-08-10_0058_tripod_stability/b2_tripod_reward_w0.3/logs/rsl_rl/crab_hex_flat_walk/2026-08-10_16-43-37/model_20998.pt
episodes   : 10  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.3884168590511688  p25=0.3639626286454573  p75=0.3932856040597608
tippy_tap_fraction median=0.0689660867195693
slip_ratio         median=0.021786855253536432
schedule_completion_rate=1.0
terminations={'schedule_complete': 10}

by hold:
       low: tripod median=0.5008083343481872 (n=10)
       mid: tripod median=0.40272665244415917 (n=10)
      high: tripod median=0.24639623880896983 (n=10)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
