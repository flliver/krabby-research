```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-21_20-14-29/model_999.pt
episodes   : 100  unscored: 52
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.0  p25=0.0  p75=0.0
tippy_tap_fraction median=0.528782894736842
slip_ratio         median=0.32613570728541513
tracking_ratio     median=0.3543730312122557  (achieved/commanded vx, walking holds; n=11)
schedule_completion_rate=0.0
terminations={'fall': 100}

by hold:
     stand: cmd 0.00 -> achieved -0.017 m/s | tripod median=0.0 (n=48)
     creep: cmd 0.25 -> achieved 0.089 m/s | tripod median=0.0 (n=9)

[WARN] num_prop=75 but observation width is 1127; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
