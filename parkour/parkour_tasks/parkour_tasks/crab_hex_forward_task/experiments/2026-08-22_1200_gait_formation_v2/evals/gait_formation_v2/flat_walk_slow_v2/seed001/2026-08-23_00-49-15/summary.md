```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-22_18-54-11/model_4999.pt
episodes   : 100  unscored: 1
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.0  p25=0.0  p75=0.004318677594924291
tippy_tap_fraction median=0.43571282795528243
slip_ratio         median=0.5735233927958339
tracking_ratio     median=0.06329251981842623  (achieved/commanded vx, walking holds; n=68)
schedule_completion_rate=0.25
terminations={'fall': 75, 'schedule_complete': 25}

by hold:
     stand: cmd 0.00 -> achieved 0.000 m/s | tripod median=0.0 (n=99)
     creep: cmd 0.25 -> achieved 0.016 m/s | tripod median=0.0 (n=68)
       low: cmd 0.35 -> achieved 0.022 m/s | tripod median=0.0 (n=54)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
