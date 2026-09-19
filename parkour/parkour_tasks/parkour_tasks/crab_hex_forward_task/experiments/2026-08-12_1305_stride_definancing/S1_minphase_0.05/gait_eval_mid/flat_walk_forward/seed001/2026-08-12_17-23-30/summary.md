```
=== crab-hex gait eval ===
scenario   : flat_walk_forward  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/2026-08-12_1305_stride_definancing/S1_minphase_0.05/logs/rsl_rl/crab_hex_flat_walk/2026-08-12_12-46-49/model_21000.pt
episodes   : 10  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.4053345906839597  p25=0.36933380004081395  p75=0.41805015602301554
tippy_tap_fraction median=0.07044064084557336
slip_ratio         median=0.023354864702986602
schedule_completion_rate=1.0
terminations={'schedule_complete': 10}

by hold:
       low: tripod median=0.5191350606261651 (n=10)
       mid: tripod median=0.418123259745081 (n=10)
      high: tripod median=0.27434358547219434 (n=10)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
