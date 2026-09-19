```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-21_18-41-10/model_999.pt
episodes   : 100  unscored: 5
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.002616482119577783  p25=0.0  p75=0.02077800381958423
tippy_tap_fraction median=0.38585607940446653
slip_ratio         median=0.36252763996657633
tracking_ratio     median=0.43142999321445014  (achieved/commanded vx, walking holds; n=8)
schedule_completion_rate=0.0
terminations={'fall': 100}

by hold:
     stand: cmd 0.00 -> achieved -0.030 m/s | tripod median=0.002452613363389909 (n=95)
     creep: cmd 0.25 -> achieved 0.108 m/s | tripod median=0.0 (n=7)

[WARN] num_prop=75 but observation width is 1127; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
