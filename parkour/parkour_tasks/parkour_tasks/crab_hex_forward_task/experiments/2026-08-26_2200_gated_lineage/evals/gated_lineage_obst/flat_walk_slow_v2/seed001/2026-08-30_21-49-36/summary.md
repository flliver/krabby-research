```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-30_15-22-57/model_14997.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.37977651304998616  p25=0.3273936505596847  p75=0.41749195774519465
tippy_tap_fraction median=0.2779866332497911
slip_ratio         median=0.30347732955542683
tracking_ratio     median=0.5683545349217521  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.66
terminations={'schedule_complete': 66, 'fall': 34}

by hold:
     stand: cmd 0.00 -> achieved 0.006 m/s | tripod median=0.09835451218695679 (n=100)
     creep: cmd 0.25 -> achieved 0.161 m/s | tripod median=0.5938604423402838 (n=100)
       low: cmd 0.35 -> achieved 0.166 m/s | tripod median=0.489872292326782 (n=78)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
