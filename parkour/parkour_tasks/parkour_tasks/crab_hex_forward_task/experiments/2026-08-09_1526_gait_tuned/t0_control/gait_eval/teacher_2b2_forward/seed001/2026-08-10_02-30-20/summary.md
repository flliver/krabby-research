```
=== crab-hex gait eval ===
scenario   : teacher_2b2_forward  (Isaac-Crab-Hex-Teacher-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/gait_tuned/t0_control/logs/rsl_rl/crab_hex_teacher/2026-08-09_22-09-26/model_20696.pt
episodes   : 10  unscored: 1
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.32537829332203433  p25=0.011384275465733505  p75=0.3676247748775178
tippy_tap_fraction median=0.2714127686472819
slip_ratio         median=0.044024275894620817
schedule_completion_rate=0.5
terminations={'fall': 5, 'schedule_complete': 5}

by hold:
       low: tripod median=0.41585262657985844 (n=7)
       mid: tripod median=0.30929544044005014 (n=7)
      high: tripod median=0.27013892332581146 (n=8)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
