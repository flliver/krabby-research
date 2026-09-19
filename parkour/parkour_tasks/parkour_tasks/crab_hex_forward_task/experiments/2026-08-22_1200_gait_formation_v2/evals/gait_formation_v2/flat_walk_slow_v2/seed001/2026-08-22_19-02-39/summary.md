```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-22_13-15-41/model_4999.pt
episodes   : 100  unscored: 13
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.0  p25=0.0  p75=0.02819535683543143
tippy_tap_fraction median=0.3642506142506142
slip_ratio         median=0.5655596999774244
tracking_ratio     median=0.008176759776207973  (achieved/commanded vx, walking holds; n=77)
schedule_completion_rate=0.67
terminations={'fall': 33, 'schedule_complete': 67}

by hold:
     stand: cmd 0.00 -> achieved -0.007 m/s | tripod median=0.0 (n=84)
     creep: cmd 0.25 -> achieved 0.002 m/s | tripod median=0.0 (n=75)
       low: cmd 0.35 -> achieved 0.002 m/s | tripod median=0.0 (n=70)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
