```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-21_17-08-24/model_999.pt
episodes   : 100  unscored: 1
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.0  p25=0.0  p75=0.0001425963815361806
tippy_tap_fraction median=0.5
slip_ratio         median=0.6534523816869098
tracking_ratio     median=0.23124557882567026  (achieved/commanded vx, walking holds; n=59)
schedule_completion_rate=0.0
terminations={'fall': 100}

by hold:
     stand: cmd 0.00 -> achieved 0.009 m/s | tripod median=0.0 (n=99)
     creep: cmd 0.25 -> achieved 0.058 m/s | tripod median=0.0 (n=57)

[WARN] num_prop=75 but observation width is 1127; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
