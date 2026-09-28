```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-29_02-35-04/model_9997.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.43474529602958595  p25=0.4080744771312279  p75=0.4520974188989889
tippy_tap_fraction median=0.2962996553600581
slip_ratio         median=0.3298655863332436
tracking_ratio     median=0.499868793472816  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.99
terminations={'schedule_complete': 99, 'fall': 1}

by hold:
     stand: cmd 0.00 -> achieved 0.010 m/s | tripod median=0.11860987552735594 (n=100)
     creep: cmd 0.25 -> achieved 0.145 m/s | tripod median=0.5946260541544415 (n=100)
       low: cmd 0.35 -> achieved 0.146 m/s | tripod median=0.5846754604031017 (n=99)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
