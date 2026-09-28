```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-21_05-52-47/model_4999.pt
episodes   : 10  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.0  p25=0.0  p75=0.0
tippy_tap_fraction median=0.1678395496129486
slip_ratio         median=0.3270419119998671
schedule_completion_rate=0.1
terminations={'fall': 9, 'schedule_complete': 1}

by hold:
     stand: tripod median=0.0 (n=10)
     creep: tripod median=0.0 (n=8)
       low: tripod median=0.0 (n=3)

[WARN] num_prop=75 but observation width is 1127; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
