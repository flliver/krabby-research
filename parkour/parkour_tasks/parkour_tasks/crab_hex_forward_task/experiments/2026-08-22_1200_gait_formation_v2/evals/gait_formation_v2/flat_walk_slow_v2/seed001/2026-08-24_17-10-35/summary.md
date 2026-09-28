```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-24_11-16-30/model_4999.pt
episodes   : 100  unscored: 9
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.021893294836849368  p25=0.008488981073384493  p75=0.05260708843975419
tippy_tap_fraction median=0.39625014625014626
slip_ratio         median=0.5003668050675215
tracking_ratio     median=0.12252727000189559  (achieved/commanded vx, walking holds; n=90)
schedule_completion_rate=0.07
terminations={'fall': 93, 'schedule_complete': 7}

by hold:
     stand: cmd 0.00 -> achieved -0.023 m/s | tripod median=0.03292405432391869 (n=91)
     creep: cmd 0.25 -> achieved 0.020 m/s | tripod median=0.004475654325778474 (n=90)
       low: cmd 0.35 -> achieved 0.053 m/s | tripod median=0.0 (n=44)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
