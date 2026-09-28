```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-21_19-28-16/model_999.pt
episodes   : 100  unscored: 8
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.0  p25=0.0  p75=0.0
tippy_tap_fraction median=0.45357142857142857
slip_ratio         median=0.49288917471376203
tracking_ratio     median=0.08800613324213596  (achieved/commanded vx, walking holds; n=86)
schedule_completion_rate=0.12
terminations={'fall': 88, 'schedule_complete': 12}

by hold:
     stand: cmd 0.00 -> achieved -0.000 m/s | tripod median=0.0 (n=78)
     creep: cmd 0.25 -> achieved 0.002 m/s | tripod median=0.0 (n=86)
       low: cmd 0.35 -> achieved 0.040 m/s | tripod median=0.0 (n=48)

[WARN] num_prop=75 but observation width is 1127; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
