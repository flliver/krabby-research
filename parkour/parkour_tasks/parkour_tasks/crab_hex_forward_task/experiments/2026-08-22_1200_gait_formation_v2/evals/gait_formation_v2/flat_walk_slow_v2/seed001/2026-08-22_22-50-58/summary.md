```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-22_16-56-30/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.0  p25=0.0  p75=0.012940107490315532
tippy_tap_fraction median=0.398366530544212
slip_ratio         median=0.5588430494778006
tracking_ratio     median=0.07503209273260185  (achieved/commanded vx, walking holds; n=94)
schedule_completion_rate=0.12
terminations={'fall': 88, 'schedule_complete': 12}

by hold:
     stand: cmd 0.00 -> achieved -0.002 m/s | tripod median=0.0 (n=100)
     creep: cmd 0.25 -> achieved 0.017 m/s | tripod median=0.0 (n=94)
       low: cmd 0.35 -> achieved 0.023 m/s | tripod median=0.0 (n=42)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
