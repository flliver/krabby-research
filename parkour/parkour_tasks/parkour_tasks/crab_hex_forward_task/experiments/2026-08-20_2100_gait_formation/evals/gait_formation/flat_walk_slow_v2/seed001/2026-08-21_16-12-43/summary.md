```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-21_00-51-42/model_999.pt
episodes   : 100  unscored: 5
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.0008735316538433634  p25=0.0  p75=0.00263567619792396
tippy_tap_fraction median=0.43021739130434783
slip_ratio         median=0.4071647287783075
schedule_completion_rate=0.12
terminations={'fall': 88, 'schedule_complete': 12}

by hold:
     stand: tripod median=0.0 (n=95)
     creep: tripod median=0.0 (n=75)
       low: tripod median=0.0 (n=29)

[WARN] num_prop=75 but observation width is 1127; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
