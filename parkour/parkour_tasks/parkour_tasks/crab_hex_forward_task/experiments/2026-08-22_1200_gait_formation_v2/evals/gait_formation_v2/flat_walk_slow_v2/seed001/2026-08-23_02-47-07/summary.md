```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-22_20-54-09/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.0  p25=0.0  p75=2.423804278520638e-05
tippy_tap_fraction median=0.5157129455909943
slip_ratio         median=0.7419495697570143
tracking_ratio     median=0.18406582718812292  (achieved/commanded vx, walking holds; n=90)
schedule_completion_rate=0.11
terminations={'fall': 89, 'schedule_complete': 11}

by hold:
     stand: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.0 (n=100)
     creep: cmd 0.25 -> achieved 0.040 m/s | tripod median=0.0 (n=87)
       low: cmd 0.35 -> achieved 0.059 m/s | tripod median=0.0 (n=30)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
