```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-31_01-12-23/model_19996.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.36691548336426505  p25=0.31704992050463243  p75=0.42444011486457317
tippy_tap_fraction median=0.29716350013379716
slip_ratio         median=0.2933605499232953
tracking_ratio     median=0.5391098932540488  (achieved/commanded vx, walking holds; n=93)
schedule_completion_rate=0.42
terminations={'schedule_complete': 42, 'fall': 58}

by hold:
     stand: cmd 0.00 -> achieved 0.013 m/s | tripod median=0.11753771852197648 (n=100)
     creep: cmd 0.25 -> achieved 0.154 m/s | tripod median=0.6052960875579545 (n=93)
       low: cmd 0.35 -> achieved 0.152 m/s | tripod median=0.46836906700406056 (n=62)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
