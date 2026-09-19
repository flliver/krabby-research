```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-28_09-42-07/model_4999.pt
episodes   : 100  unscored: 5
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.0  p25=0.0  p75=0.0029622524135311306
tippy_tap_fraction median=0.2561930783242259
slip_ratio         median=1.408411513936872
tracking_ratio     median=0.014773412146466072  (achieved/commanded vx, walking holds; n=94)
schedule_completion_rate=0.91
terminations={'schedule_complete': 91, 'fall': 9}

by hold:
     stand: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.0 (n=95)
     creep: cmd 0.25 -> achieved 0.001 m/s | tripod median=0.0 (n=94)
       low: cmd 0.35 -> achieved 0.009 m/s | tripod median=0.0 (n=94)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
