```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_side  (Isaac-Crab-Hex-Side-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_side_teacher/2026-09-11_14-50-01/model_9998.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.3722502413135953  p25=0.32912641032972634  p75=0.415195094424616
tippy_tap_fraction median=0.3889577814102926
slip_ratio         median=0.4080983995007108
tracking_ratio     median=0.5986844964241449  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.94
terminations={'schedule_complete': 94, 'fall': 6}

by hold:
     stand: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.0 (n=92)
     creep: cmd 0.25 -> achieved 0.189 m/s | tripod median=0.5817889698298306 (n=100)
       low: cmd 0.35 -> achieved 0.163 m/s | tripod median=0.5227093762895305 (n=99)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
