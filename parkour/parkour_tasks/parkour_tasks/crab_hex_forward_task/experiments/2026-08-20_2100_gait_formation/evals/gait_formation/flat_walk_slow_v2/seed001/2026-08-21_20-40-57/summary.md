```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-21_14-55-08/model_9998.pt
episodes   : 100  unscored: 2
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.0  p25=0.0  p75=0.0
tippy_tap_fraction median=0.14258962011770998
slip_ratio         median=0.2868796278884404
tracking_ratio     median=0.09417419289639298  (achieved/commanded vx, walking holds; n=84)
schedule_completion_rate=0.38
terminations={'fall': 62, 'schedule_complete': 38}

by hold:
     stand: cmd 0.00 -> achieved 0.025 m/s | tripod median=0.0 (n=98)
     creep: cmd 0.25 -> achieved 0.024 m/s | tripod median=0.0 (n=84)
       low: cmd 0.35 -> achieved 0.030 m/s | tripod median=0.0 (n=70)

[WARN] num_prop=75 but observation width is 1127; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
