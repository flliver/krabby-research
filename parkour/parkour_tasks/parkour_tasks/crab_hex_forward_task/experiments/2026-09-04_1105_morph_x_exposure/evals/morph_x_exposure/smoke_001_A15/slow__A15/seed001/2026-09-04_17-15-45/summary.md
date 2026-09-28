```
=== crab-hex gait eval ===
scenario   : slow__A15  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-04_13-09-45/model_199.pt
episodes   : 100  unscored: 7
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.4349352088516926  p25=0.40161784097279435  p75=0.46740950756985017
tippy_tap_fraction median=0.3170713938805245
slip_ratio         median=0.23761603598913755
tracking_ratio     median=0.6339974397464281  (achieved/commanded vx, walking holds; n=82)
schedule_completion_rate=0.79
terminations={'fall': 21, 'schedule_complete': 79}

by hold:
     stand: cmd 0.00 -> achieved 0.083 m/s | tripod median=0.28812234469059494 (n=93)
     creep: cmd 0.25 -> achieved 0.184 m/s | tripod median=0.5647284808997901 (n=82)
       low: cmd 0.35 -> achieved 0.183 m/s | tripod median=0.48017974993410845 (n=81)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
