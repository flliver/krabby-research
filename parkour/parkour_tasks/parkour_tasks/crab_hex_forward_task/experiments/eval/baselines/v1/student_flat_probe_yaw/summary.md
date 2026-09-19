```
=== crab-hex gait eval ===
scenario   : student_flat_probe_yaw  (Isaac-Crab-Hex-Student-v0)
checkpoint : logs/rsl_rl/crab_hex_student/2026-08-06_14-22-48/model_29098.pt
episodes   : 5  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.06898757137416724  p25=0.036749459722744314  p75=0.07362831439535526
tippy_tap_fraction median=0.3
slip_ratio         median=0.06310763851453788
schedule_completion_rate=1.0
terminations={'schedule_complete': 5}

by hold:
  straight: tripod median=0.08062893742984195 (n=5)
   yaw_pos: tripod median=0.05219593544889811 (n=5)
   yaw_neg: tripod median=0.04786353634285985 (n=5)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
