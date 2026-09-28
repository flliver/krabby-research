```
=== crab-hex gait eval ===
scenario   : flat_walk_forward  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/short_runs/r03_airtime_w0.8/logs/rsl_rl/crab_hex_flat_walk/2026-08-09_10-35-18/model_20998.pt
episodes   : 10  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.0  p25=0.0  p75=0.0
tippy_tap_fraction median=0.1801998569689004
slip_ratio         median=0.015916526647210667
schedule_completion_rate=1.0
terminations={'schedule_complete': 10}

by hold:
       low: tripod median=0.0 (n=10)
       mid: tripod median=0.0 (n=10)
      high: tripod median=0.0 (n=10)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
