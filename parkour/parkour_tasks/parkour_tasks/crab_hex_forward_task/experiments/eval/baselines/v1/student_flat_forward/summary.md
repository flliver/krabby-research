```
=== crab-hex gait eval ===
scenario   : student_flat_forward  (Isaac-Crab-Hex-Student-v0)
checkpoint : logs/rsl_rl/crab_hex_student/2026-08-06_14-22-48/model_29098.pt
episodes   : 10  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.08021908356186273  p25=0.02569913287075576  p75=0.105389745358494
tippy_tap_fraction median=0.2876124987359693
slip_ratio         median=0.08262031393412715
schedule_completion_rate=0.8
terminations={'schedule_complete': 8, 'fall': 2}

by hold:
       low: tripod median=0.06878348833885252 (n=10)
       mid: tripod median=0.10699119751096947 (n=8)
      high: tripod median=0.09425098024631137 (n=8)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
