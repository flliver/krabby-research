```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-30_15-22-57/model_14997.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.41383902856563437  p25=0.38354804537220805  p75=0.4412294641615633
tippy_tap_fraction median=0.28671568627450983
slip_ratio         median=0.2978024176747363
tracking_ratio     median=0.5636194862819841  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.99
terminations={'schedule_complete': 99, 'fall': 1}

by hold:
     stand: cmd 0.00 -> achieved 0.007 m/s | tripod median=0.10568405830803929 (n=100)
     creep: cmd 0.25 -> achieved 0.155 m/s | tripod median=0.6133126245866917 (n=100)
       low: cmd 0.35 -> achieved 0.175 m/s | tripod median=0.5248171187277751 (n=99)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
