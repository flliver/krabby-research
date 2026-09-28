```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-07_04-38-50/model_19996.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.4289731929140015  p25=0.3758099287815257  p75=0.4893270906313214
tippy_tap_fraction median=0.25
slip_ratio         median=0.19575458105360627
tracking_ratio     median=0.5670839108115349  (achieved/commanded vx, walking holds; n=90)
schedule_completion_rate=0.45
terminations={'fall': 55, 'schedule_complete': 45}

by hold:
     stand: cmd 0.00 -> achieved 0.111 m/s | tripod median=0.4454153534141831 (n=100)
     creep: cmd 0.25 -> achieved 0.158 m/s | tripod median=0.45572423659322314 (n=88)
       low: cmd 0.35 -> achieved 0.164 m/s | tripod median=0.40302209672063977 (n=57)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
