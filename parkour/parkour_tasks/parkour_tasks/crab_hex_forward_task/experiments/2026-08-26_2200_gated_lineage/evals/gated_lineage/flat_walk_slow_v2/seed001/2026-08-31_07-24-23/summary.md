```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-31_01-12-23/model_19996.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.43617479954199057  p25=0.4098919760593237  p75=0.45820300575468365
tippy_tap_fraction median=0.29349206349206347
slip_ratio         median=0.29862019717856503
tracking_ratio     median=0.4945130159443393  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.99
terminations={'schedule_complete': 99, 'fall': 1}

by hold:
     stand: cmd 0.00 -> achieved 0.010 m/s | tripod median=0.05631406769369138 (n=100)
     creep: cmd 0.25 -> achieved 0.139 m/s | tripod median=0.6289395457261251 (n=100)
       low: cmd 0.35 -> achieved 0.155 m/s | tripod median=0.6290620847691892 (n=99)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
