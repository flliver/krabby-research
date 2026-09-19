```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-21_18-18-00/model_999.pt
episodes   : 100  unscored: 9
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.0  p25=0.0  p75=0.008997124267103269
tippy_tap_fraction median=0.40588235294117647
slip_ratio         median=0.5864907615785855
tracking_ratio     median=0.14692964613011814  (achieved/commanded vx, walking holds; n=74)
schedule_completion_rate=0.02
terminations={'fall': 98, 'schedule_complete': 2}

by hold:
     stand: cmd 0.00 -> achieved -0.001 m/s | tripod median=0.0 (n=87)
     creep: cmd 0.25 -> achieved 0.037 m/s | tripod median=0.0 (n=68)
       low: cmd 0.35 -> achieved 0.001 m/s | tripod median=0.005038910519697702 (n=2)

[WARN] num_prop=75 but observation width is 1127; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
