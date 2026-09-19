```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-31_06-04-16/model_9998.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5992655249432874  p25=0.5661422749431788  p75=0.6409112602923493
tippy_tap_fraction median=0.22188955422488355
slip_ratio         median=0.19940715588009267
tracking_ratio     median=0.41326730739807027  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.99
terminations={'schedule_complete': 99, 'fall': 1}

by hold:
     stand: cmd 0.00 -> achieved 0.044 m/s | tripod median=0.4972421848119977 (n=100)
     creep: cmd 0.25 -> achieved 0.109 m/s | tripod median=0.6488575795832223 (n=100)
       low: cmd 0.35 -> achieved 0.136 m/s | tripod median=0.6785337353564769 (n=99)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
