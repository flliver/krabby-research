```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-29_00-37-47/model_6998.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.44970102221800456  p25=0.4245060268139157  p75=0.46648384174624064
tippy_tap_fraction median=0.26578366445916113
slip_ratio         median=0.294032814198979
tracking_ratio     median=0.4549774929239505  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}

by hold:
     stand: cmd 0.00 -> achieved 0.006 m/s | tripod median=0.15045449722850934 (n=100)
     creep: cmd 0.25 -> achieved 0.128 m/s | tripod median=0.5831715879545567 (n=100)
       low: cmd 0.35 -> achieved 0.140 m/s | tripod median=0.609723323957443 (n=100)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
