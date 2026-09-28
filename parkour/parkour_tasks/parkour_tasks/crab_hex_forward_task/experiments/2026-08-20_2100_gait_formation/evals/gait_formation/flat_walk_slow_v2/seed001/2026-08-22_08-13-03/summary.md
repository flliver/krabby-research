```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-22_02-27-29/model_4999.pt
episodes   : 100  unscored: 2
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.0  p25=0.0  p75=0.001067765054388308
tippy_tap_fraction median=0.39214847759151555
slip_ratio         median=0.469103365333913
tracking_ratio     median=0.7419618071625697  (achieved/commanded vx, walking holds; n=75)
schedule_completion_rate=0.05
terminations={'fall': 95, 'schedule_complete': 5}

by hold:
     stand: cmd 0.00 -> achieved 0.020 m/s | tripod median=0.0 (n=98)
     creep: cmd 0.25 -> achieved 0.183 m/s | tripod median=0.0 (n=75)
       low: cmd 0.35 -> achieved 0.209 m/s | tripod median=0.0 (n=21)

[WARN] num_prop=75 but observation width is 1127; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
