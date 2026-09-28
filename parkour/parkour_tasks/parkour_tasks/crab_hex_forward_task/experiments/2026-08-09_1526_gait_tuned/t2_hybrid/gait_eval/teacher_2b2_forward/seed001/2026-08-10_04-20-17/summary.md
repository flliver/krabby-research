```
=== crab-hex gait eval ===
scenario   : teacher_2b2_forward  (Isaac-Crab-Hex-Teacher-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/gait_tuned/t2_hybrid/logs/rsl_rl/crab_hex_teacher/2026-08-09_23-51-54/model_21694.pt
episodes   : 10  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.18453239128457338  p25=0.02848301211597145  p75=0.35409392952286917
tippy_tap_fraction median=0.3464285714285714
slip_ratio         median=0.05602634644858718
schedule_completion_rate=0.6
terminations={'fall': 4, 'schedule_complete': 6}

by hold:
       low: tripod median=0.4037965423145976 (n=8)
       mid: tripod median=0.262108572779171 (n=7)
      high: tripod median=0.25717202923537374 (n=7)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
