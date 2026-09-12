```
=== crab-hex gait eval ===
scenario   : step__side  (Isaac-Crab-Hex-Side-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_side_teacher/2026-09-12_03-04-37/model_9800.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.36621668488515907  p25=0.2997969561858266  p75=0.42515369696219885
tippy_tap_fraction median=0.3830478032605692
slip_ratio         median=0.5031638476481767
tracking_ratio     median=0.5709953508390863  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.86
terminations={'schedule_complete': 86, 'fall': 14}

by hold:
     stand: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.0 (n=91)
     creep: cmd 0.25 -> achieved 0.186 m/s | tripod median=0.6098395501261749 (n=100)
       low: cmd 0.35 -> achieved 0.141 m/s | tripod median=0.48422190587294023 (n=94)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
