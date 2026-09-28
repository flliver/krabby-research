```
=== crab-hex gait eval ===
scenario   : flat_walk_forward  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/2026-08-10_0058_tripod_stability/b7_tripod_v5_w0.3_pitch-0.1/logs/rsl_rl/crab_hex_flat_walk/2026-08-12_04-59-42/model_24997.pt
episodes   : 10  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.42484623631995155  p25=0.40682675779362015  p75=0.4300676519763133
tippy_tap_fraction median=0.07173734046932895
slip_ratio         median=0.022952204393412647
schedule_completion_rate=1.0
terminations={'schedule_complete': 10}

by hold:
       low: tripod median=0.5196278642356679 (n=10)
       mid: tripod median=0.4250917437767303 (n=10)
      high: tripod median=0.29059192043409593 (n=10)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
