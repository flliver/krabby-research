```
=== crab-hex gait eval ===
scenario   : flat_walk_forward  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/2026-08-10_0058_tripod_stability/b6b_tripod_v5_ft_w0.3/logs/rsl_rl/crab_hex_flat_walk/2026-08-12_03-06-23/model_25997.pt
episodes   : 10  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.405435979186326  p25=0.3933690272173996  p75=0.4239606268381356
tippy_tap_fraction median=0.0691066938281822
slip_ratio         median=0.021989838634237542
schedule_completion_rate=1.0
terminations={'schedule_complete': 10}

by hold:
       low: tripod median=0.5069267475260792 (n=10)
       mid: tripod median=0.4189468492705436 (n=10)
      high: tripod median=0.2891794024014972 (n=10)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
