```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-23_20-08-24/model_14997.pt
episodes   : 100  unscored: 4
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5426188119298889  p25=0.5143494867571561  p75=0.5787516899006545
tippy_tap_fraction median=0.22674418604651161
slip_ratio         median=0.24150149876884433
tracking_ratio     median=0.4865453945724838  (achieved/commanded vx, walking holds; n=96)
schedule_completion_rate=0.9
terminations={'schedule_complete': 90, 'fall': 10}

by hold:
     stand: cmd 0.00 -> achieved 0.027 m/s | tripod median=0.459882210509857 (n=96)
     creep: cmd 0.25 -> achieved 0.148 m/s | tripod median=0.6295814622324039 (n=96)
       low: cmd 0.35 -> achieved 0.131 m/s | tripod median=0.5576833125527766 (n=93)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
