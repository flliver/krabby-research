```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-01_04-52-20/model_11997.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6073424072628063  p25=0.5723406633043322  p75=0.6333316692925348
tippy_tap_fraction median=0.24715578539107952
slip_ratio         median=0.2044570423951425
tracking_ratio     median=0.46676928000034834  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.99
terminations={'schedule_complete': 99, 'fall': 1}

by hold:
     stand: cmd 0.00 -> achieved 0.053 m/s | tripod median=0.5890647270893365 (n=100)
     creep: cmd 0.25 -> achieved 0.122 m/s | tripod median=0.6424596885041198 (n=100)
       low: cmd 0.35 -> achieved 0.152 m/s | tripod median=0.6056024931771461 (n=100)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
