```
=== crab-hex gait eval ===
scenario   : step__A20  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-31_10-46-19/model_19996.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6158752940101389  p25=0.5733796044372074  p75=0.6681466183023294
tippy_tap_fraction median=0.2358288770053476
slip_ratio         median=0.2156541050827807
tracking_ratio     median=0.39059097763475487  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}

by hold:
     stand: cmd 0.00 -> achieved 0.064 m/s | tripod median=0.6671097520692539 (n=100)
     creep: cmd 0.25 -> achieved 0.101 m/s | tripod median=0.6305695139707839 (n=100)
       low: cmd 0.35 -> achieved 0.126 m/s | tripod median=0.6022615796643536 (n=100)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
