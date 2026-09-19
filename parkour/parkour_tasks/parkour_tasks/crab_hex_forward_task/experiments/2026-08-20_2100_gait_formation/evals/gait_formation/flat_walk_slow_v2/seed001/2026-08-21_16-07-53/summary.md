```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-21_11-45-42/model_999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.010306910713071646  p25=0.0011146513657067133  p75=0.02373440951860769
tippy_tap_fraction median=0.42396260829913357
slip_ratio         median=0.5347406347745218
schedule_completion_rate=0.0
terminations={'fall': 100}

by hold:
     stand: tripod median=0.0 (n=100)
     creep: tripod median=0.027563321108585993 (n=87)
       low: tripod median=0.010983418128655826 (n=18)

[WARN] num_prop=75 but observation width is 1127; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
