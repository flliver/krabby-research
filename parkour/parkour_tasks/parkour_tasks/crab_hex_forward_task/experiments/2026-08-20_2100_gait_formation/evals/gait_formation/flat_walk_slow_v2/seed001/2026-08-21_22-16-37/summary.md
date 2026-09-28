```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-21_17-54-37/model_999.pt
episodes   : 100  unscored: 3
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.0  p25=0.0  p75=0.0
tippy_tap_fraction median=0.20633768746976294
slip_ratio         median=0.4441970635198108
tracking_ratio     median=0.2158054678694777  (achieved/commanded vx, walking holds; n=33)
schedule_completion_rate=0.0
terminations={'fall': 100}

by hold:
     stand: cmd 0.00 -> achieved -0.021 m/s | tripod median=0.0 (n=97)
     creep: cmd 0.25 -> achieved 0.051 m/s | tripod median=0.0 (n=32)
       low: cmd 0.35 -> achieved 0.078 m/s | tripod median=0.0 (n=6)

[WARN] num_prop=75 but observation width is 1127; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
