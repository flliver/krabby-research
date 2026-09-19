```
=== crab-hex gait eval ===
scenario   : student_2b2_forward  (Isaac-Crab-Hex-Student-v0)
checkpoint : logs/rsl_rl/crab_hex_student/2026-08-06_14-22-48/model_29098.pt
episodes   : 10  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.07852714869825159  p25=0.03172131663178217  p75=0.10803774632188048
tippy_tap_fraction median=0.2972433549356626
slip_ratio         median=0.07172472973806072
schedule_completion_rate=0.9
terminations={'schedule_complete': 9, 'fall': 1}

by hold:
       low: tripod median=0.04521598896742808 (n=10)
       mid: tripod median=0.09257672224340548 (n=9)
      high: tripod median=0.12951333949027138 (n=9)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
