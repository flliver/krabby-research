```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-21_12-17-35/model_9998.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.0  p25=0.0  p75=0.0
tippy_tap_fraction median=0.34066731141199225
slip_ratio         median=0.5476238296834761
schedule_completion_rate=0.92
terminations={'schedule_complete': 92, 'fall': 8}

by hold:
     stand: tripod median=0.0 (n=100)
     creep: tripod median=0.0 (n=100)
       low: tripod median=0.0 (n=100)

[WARN] num_prop=75 but observation width is 1127; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
