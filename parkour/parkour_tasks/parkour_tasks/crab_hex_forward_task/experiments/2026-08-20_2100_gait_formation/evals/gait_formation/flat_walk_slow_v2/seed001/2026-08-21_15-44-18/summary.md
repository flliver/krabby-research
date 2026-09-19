```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-21_11-22-17/model_999.pt
episodes   : 100  unscored: 5
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.0  p25=0.0  p75=0.0001699024903339735
tippy_tap_fraction median=0.4588963963963964
slip_ratio         median=0.5677926389568357
schedule_completion_rate=0.0
terminations={'fall': 100}

by hold:
     stand: tripod median=0.0 (n=95)
     creep: tripod median=0.0 (n=71)
       low: tripod median=0.0 (n=5)

[WARN] num_prop=75 but observation width is 1127; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
