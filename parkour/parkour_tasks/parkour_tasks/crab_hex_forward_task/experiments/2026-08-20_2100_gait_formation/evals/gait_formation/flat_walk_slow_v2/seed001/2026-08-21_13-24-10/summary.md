```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-21_07-36-47/model_4999.pt
episodes   : 10  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.0  p25=0.0  p75=0.0006345363438984919
tippy_tap_fraction median=0.3039136302294197
slip_ratio         median=0.410366979622479
schedule_completion_rate=0.4
terminations={'fall': 6, 'schedule_complete': 4}

by hold:
     stand: tripod median=0.0 (n=10)
     creep: tripod median=0.0 (n=10)
       low: tripod median=0.0 (n=9)

[WARN] num_prop=75 but observation width is 1127; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
