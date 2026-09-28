```
=== crab-hex gait eval ===
scenario   : flat_walk_forward  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/2026-08-12_1017_lean_reduction/L5_fwdprog_0.3/logs/rsl_rl/crab_hex_flat_walk/2026-08-12_11-35-05/model_21000.pt
episodes   : 10  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.39717508652901956  p25=0.38877163720477664  p75=0.40488406381363445
tippy_tap_fraction median=0.06188479189745213
slip_ratio         median=0.0238528843796184
schedule_completion_rate=1.0
terminations={'schedule_complete': 10}

by hold:
       low: tripod median=0.4663264614813276 (n=10)
       mid: tripod median=0.4267332718771799 (n=10)
      high: tripod median=0.30305321440639615 (n=10)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
