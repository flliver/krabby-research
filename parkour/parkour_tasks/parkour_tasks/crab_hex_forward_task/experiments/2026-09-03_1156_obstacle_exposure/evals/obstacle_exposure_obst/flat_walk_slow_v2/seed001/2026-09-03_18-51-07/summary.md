```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-01_02-22-26/model_9998.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5342346326338664  p25=0.4882266181187635  p75=0.5890737678042939
tippy_tap_fraction median=0.23513452770821291
slip_ratio         median=0.2324642854371663
tracking_ratio     median=0.41543359495082743  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.8
terminations={'schedule_complete': 80, 'fall': 20}

by hold:
     stand: cmd 0.00 -> achieved 0.039 m/s | tripod median=0.4885481530995719 (n=100)
     creep: cmd 0.25 -> achieved 0.108 m/s | tripod median=0.6255393965859741 (n=100)
       low: cmd 0.35 -> achieved 0.140 m/s | tripod median=0.5208619597276308 (n=93)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
