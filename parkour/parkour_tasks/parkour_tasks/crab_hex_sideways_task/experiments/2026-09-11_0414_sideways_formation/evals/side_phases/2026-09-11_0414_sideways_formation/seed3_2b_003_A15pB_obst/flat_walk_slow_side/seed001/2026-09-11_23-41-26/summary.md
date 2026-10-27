```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_side  (Isaac-Crab-Hex-Side-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_side_teacher/2026-09-11_17-19-40/model_14997.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.36364997241858776  p25=0.31217075323549554  p75=0.4052685062461867
tippy_tap_fraction median=0.40724352671255326
slip_ratio         median=0.40108121922695883
tracking_ratio     median=0.6073520113188733  (achieved/commanded vx, walking holds; n=99)
schedule_completion_rate=0.87
terminations={'schedule_complete': 87, 'fall': 13}

by hold:
     stand: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.0 (n=91)
     creep: cmd 0.25 -> achieved 0.189 m/s | tripod median=0.58959609726162 (n=99)
       low: cmd 0.35 -> achieved 0.165 m/s | tripod median=0.4854777093576234 (n=92)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
