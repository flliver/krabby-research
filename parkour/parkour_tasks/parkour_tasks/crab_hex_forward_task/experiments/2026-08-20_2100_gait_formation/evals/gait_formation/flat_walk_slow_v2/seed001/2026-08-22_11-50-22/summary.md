```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-22_06-03-19/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.0  p25=0.0  p75=0.0
tippy_tap_fraction median=0.43978021978021975
slip_ratio         median=0.5449149601997403
tracking_ratio     median=0.4744359941934502  (achieved/commanded vx, walking holds; n=70)
schedule_completion_rate=0.0
terminations={'fall': 100}

by hold:
     stand: cmd 0.00 -> achieved -0.003 m/s | tripod median=0.0 (n=100)
     creep: cmd 0.25 -> achieved 0.119 m/s | tripod median=0.0 (n=66)

[WARN] num_prop=75 but observation width is 1127; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
