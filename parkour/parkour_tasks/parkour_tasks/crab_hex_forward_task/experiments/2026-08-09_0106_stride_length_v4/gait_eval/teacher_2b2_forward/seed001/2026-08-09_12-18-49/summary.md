```
=== crab-hex gait eval ===
scenario   : teacher_2b2_forward  (Isaac-Crab-Hex-Teacher-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/stride_length_v4/logs/rsl_rl/crab_hex_teacher/2026-08-09_08-03-07/model_22000.pt
episodes   : 10  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.004956390373630562  p25=0.000623511986328778  p75=0.007500088193765919
tippy_tap_fraction median=0.2873685007189889
slip_ratio         median=0.04333757108605064
schedule_completion_rate=0.8
terminations={'schedule_complete': 8, 'fall': 2}

by hold:
       low: tripod median=0.0019237993037415967 (n=9)
       mid: tripod median=0.0006087076095564738 (n=8)
      high: tripod median=0.0 (n=9)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
