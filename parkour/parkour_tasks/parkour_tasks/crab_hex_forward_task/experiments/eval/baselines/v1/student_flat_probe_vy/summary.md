```
=== crab-hex gait eval ===
scenario   : student_flat_probe_vy  (Isaac-Crab-Hex-Student-v0)
checkpoint : logs/rsl_rl/crab_hex_student/2026-08-06_14-22-48/model_29098.pt
episodes   : 5  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.054481383216853106  p25=0.010532555133046814  p75=0.09631254265058486
tippy_tap_fraction median=0.2798165137614679
slip_ratio         median=0.0536509244312999
schedule_completion_rate=1.0
terminations={'schedule_complete': 5}

by hold:
  baseline: tripod median=0.027220475630255328 (n=5)
    vy_pos: tripod median=0.0745643462771748 (n=5)
    vy_neg: tripod median=0.0616593277431292 (n=5)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
