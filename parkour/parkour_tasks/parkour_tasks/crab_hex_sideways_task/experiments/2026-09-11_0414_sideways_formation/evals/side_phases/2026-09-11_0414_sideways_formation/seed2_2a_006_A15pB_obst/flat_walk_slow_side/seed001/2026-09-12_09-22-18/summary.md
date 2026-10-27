```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_side  (Isaac-Crab-Hex-Side-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_side_teacher/2026-09-12_03-04-37/model_9998.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.4334483425772402  p25=0.3897511469864117  p75=0.47809810047539275
tippy_tap_fraction median=0.366266908380979
slip_ratio         median=0.4423950789759912
tracking_ratio     median=0.6957321166001511  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.97
terminations={'schedule_complete': 97, 'fall': 3}

by hold:
     stand: cmd 0.00 -> achieved 0.005 m/s | tripod median=0.0 (n=92)
     creep: cmd 0.25 -> achieved 0.206 m/s | tripod median=0.6404611417014683 (n=100)
       low: cmd 0.35 -> achieved 0.200 m/s | tripod median=0.6440132949046988 (n=98)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
