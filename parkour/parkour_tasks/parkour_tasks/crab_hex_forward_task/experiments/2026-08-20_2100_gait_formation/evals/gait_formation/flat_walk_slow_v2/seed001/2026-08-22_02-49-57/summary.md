```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-21_21-01-15/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.0  p25=0.0  p75=0.0
tippy_tap_fraction median=0.2782991672005125
slip_ratio         median=0.4352740420074316
tracking_ratio     median=0.3244960432315377  (achieved/commanded vx, walking holds; n=90)
schedule_completion_rate=0.25
terminations={'fall': 75, 'schedule_complete': 25}

by hold:
     stand: cmd 0.00 -> achieved -0.003 m/s | tripod median=0.0 (n=100)
     creep: cmd 0.25 -> achieved 0.084 m/s | tripod median=0.0 (n=89)
       low: cmd 0.35 -> achieved 0.103 m/s | tripod median=0.0 (n=50)

[WARN] num_prop=75 but observation width is 1127; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
