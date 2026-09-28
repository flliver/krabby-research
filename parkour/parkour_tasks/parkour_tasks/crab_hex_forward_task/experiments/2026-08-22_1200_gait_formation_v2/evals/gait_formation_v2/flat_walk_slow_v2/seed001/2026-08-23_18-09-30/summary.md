```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-23_12-16-28/model_4999.pt
episodes   : 100  unscored: 8
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.0  p25=0.0  p75=0.0
tippy_tap_fraction median=0.22921634401648952
slip_ratio         median=0.18206984000699264
tracking_ratio     median=0.42609796083499935  (achieved/commanded vx, walking holds; n=81)
schedule_completion_rate=0.12
terminations={'fall': 88, 'schedule_complete': 12}

by hold:
     stand: cmd 0.00 -> achieved 0.012 m/s | tripod median=0.0 (n=92)
     creep: cmd 0.25 -> achieved 0.107 m/s | tripod median=0.0 (n=81)
       low: cmd 0.35 -> achieved 0.125 m/s | tripod median=0.0 (n=37)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
