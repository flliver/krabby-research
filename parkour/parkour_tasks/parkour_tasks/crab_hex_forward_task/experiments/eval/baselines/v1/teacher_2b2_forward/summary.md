```
=== crab-hex gait eval ===
scenario   : teacher_2b2_forward  (Isaac-Crab-Hex-Teacher-v0)
checkpoint : logs/rsl_rl/crab_hex_teacher/2026-08-05_17-36-56/model_21100.pt
episodes   : 10  unscored: 1
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.052065444044282184  p25=0.02035685251015767  p75=0.07782137552942311
tippy_tap_fraction median=0.39220799747115537
slip_ratio         median=0.09736157327198314
schedule_completion_rate=0.7
terminations={'fall': 3, 'schedule_complete': 7}

by hold:
       low: tripod median=0.03903034894160923 (n=8)
       mid: tripod median=0.04826727051416885 (n=9)
      high: tripod median=0.06096353327996466 (n=8)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
