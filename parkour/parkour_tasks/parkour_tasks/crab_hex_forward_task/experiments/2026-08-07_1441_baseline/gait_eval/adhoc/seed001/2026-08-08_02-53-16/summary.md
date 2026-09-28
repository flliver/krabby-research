```
=== crab-hex gait eval ===
scenario   : adhoc  (Isaac-Crab-Hex-Teacher-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/baseline/logs/rsl_rl/crab_hex_teacher/2026-08-07_22-21-05/model_21500.pt
episodes   : 10  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.0035100897756172623  p25=0.0006081204958187328  p75=0.0055101793083709395
tippy_tap_fraction median=0.1471238632280178
slip_ratio         median=0.07303146901293521
schedule_completion_rate=0.9
terminations={'schedule_complete': 9, 'fall': 1}

by hold:
       low: tripod median=0.0 (n=10)
       mid: tripod median=0.002262284175199541 (n=10)
      high: tripod median=0.0 (n=9)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
