```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-21_17-31-29/model_999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.0  p25=0.0  p75=0.0
tippy_tap_fraction median=0.34193870752831446
slip_ratio         median=0.5184808337187548
tracking_ratio     median=0.10348915986469365  (achieved/commanded vx, walking holds; n=83)
schedule_completion_rate=0.12
terminations={'fall': 88, 'schedule_complete': 12}

by hold:
     stand: cmd 0.00 -> achieved -0.009 m/s | tripod median=0.0 (n=100)
     creep: cmd 0.25 -> achieved 0.020 m/s | tripod median=0.0 (n=81)
       low: cmd 0.35 -> achieved 0.049 m/s | tripod median=0.0 (n=43)

[WARN] num_prop=75 but observation width is 1127; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
