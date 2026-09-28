```
=== crab-hex gait eval ===
scenario   : flat_walk_forward  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/2026-08-09_1526_gait_tuned/logs/rsl_rl/crab_hex_flat_walk/2026-08-09_15-27-14/model_5000.pt
episodes   : 10  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.34309246344385813  p25=0.33952800301484776  p75=0.35667198147904156
tippy_tap_fraction median=0.07580239358085135
slip_ratio         median=0.024850231161249618
schedule_completion_rate=1.0
terminations={'schedule_complete': 10}

by hold:
       low: tripod median=0.37396039053887287 (n=10)
       mid: tripod median=0.3746222313946082 (n=10)
      high: tripod median=0.3103128332859313 (n=10)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
