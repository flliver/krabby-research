```
=== crab-hex gait eval ===
scenario   : step__side  (Isaac-Crab-Hex-Side-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_side_teacher/2026-09-12_05-24-07/model_14800.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.3295472047366841  p25=0.244684446123528  p75=0.390132298701954
tippy_tap_fraction median=0.3863786825554452
slip_ratio         median=0.4464246427417858
tracking_ratio     median=0.5719109476767438  (achieved/commanded vx, walking holds; n=99)
schedule_completion_rate=0.81
terminations={'schedule_complete': 81, 'fall': 19}

by hold:
     stand: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.0 (n=88)
     creep: cmd 0.25 -> achieved 0.191 m/s | tripod median=0.5509675245744607 (n=99)
       low: cmd 0.35 -> achieved 0.139 m/s | tripod median=0.429543802663494 (n=90)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
