```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-22_00-41-24/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.0  p25=0.0  p75=0.0
tippy_tap_fraction median=0.3677139761646804
slip_ratio         median=0.5311099873388927
tracking_ratio     median=0.57336639504944  (achieved/commanded vx, walking holds; n=88)
schedule_completion_rate=0.0
terminations={'fall': 100}

by hold:
     stand: cmd 0.00 -> achieved 0.000 m/s | tripod median=0.0 (n=100)
     creep: cmd 0.25 -> achieved 0.143 m/s | tripod median=0.0 (n=80)
       low: cmd 0.35 -> achieved 0.208 m/s | tripod median=0.0 (n=7)

[WARN] num_prop=75 but observation width is 1127; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
