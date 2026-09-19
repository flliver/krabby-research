```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-03_15-41-21/model_24995.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5210931716836467  p25=0.48089095321327563  p75=0.5539496544157447
tippy_tap_fraction median=0.23792873792873792
slip_ratio         median=0.2581747760834564
tracking_ratio     median=0.45936684739097267  (achieved/commanded vx, walking holds; n=99)
schedule_completion_rate=0.42
terminations={'schedule_complete': 42, 'fall': 58}

by hold:
     stand: cmd 0.00 -> achieved 0.064 m/s | tripod median=0.5908606012747144 (n=100)
     creep: cmd 0.25 -> achieved 0.119 m/s | tripod median=0.5339740414697669 (n=99)
       low: cmd 0.35 -> achieved 0.134 m/s | tripod median=0.38276302065994106 (n=67)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
