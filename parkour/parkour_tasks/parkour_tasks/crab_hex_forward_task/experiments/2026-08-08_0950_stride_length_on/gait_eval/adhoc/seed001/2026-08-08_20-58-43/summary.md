```
=== crab-hex gait eval ===
scenario   : adhoc  (Isaac-Crab-Hex-Teacher-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/stride_length_on/logs/rsl_rl/crab_hex_teacher/2026-08-08_16-27-00/model_21300.pt
episodes   : 10  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.0011553839612688777  p25=0.0  p75=0.00665933005151338
tippy_tap_fraction median=0.24495751437644836
slip_ratio         median=0.046784486065805056
schedule_completion_rate=0.6
terminations={'fall': 4, 'schedule_complete': 6}

by hold:
       low: tripod median=0.0 (n=10)
       mid: tripod median=0.0 (n=9)
      high: tripod median=0.0 (n=8)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
