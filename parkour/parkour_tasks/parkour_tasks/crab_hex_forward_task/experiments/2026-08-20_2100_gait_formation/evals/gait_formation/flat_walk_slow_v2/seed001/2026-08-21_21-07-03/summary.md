```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-21_16-45-35/model_999.pt
episodes   : 100  unscored: 5
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.0  p25=0.0  p75=0.0
tippy_tap_fraction median=0.2777777777777778
slip_ratio         median=0.32083505611152874
tracking_ratio     median=0.13804006740596117  (achieved/commanded vx, walking holds; n=93)
schedule_completion_rate=0.01
terminations={'fall': 99, 'schedule_complete': 1}

by hold:
     stand: cmd 0.00 -> achieved -0.013 m/s | tripod median=0.0 (n=95)
     creep: cmd 0.25 -> achieved 0.032 m/s | tripod median=0.0 (n=92)
       low: cmd 0.35 -> achieved 0.078 m/s | tripod median=0.0 (n=9)

[WARN] num_prop=75 but observation width is 1127; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
