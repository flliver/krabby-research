```
=== crab-hex gait eval ===
scenario   : flat_walk_forward  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/2026-08-12_1305_stride_definancing/S3_combo/logs/rsl_rl/crab_hex_flat_walk/2026-08-12_14-01-08/model_21000.pt
episodes   : 10  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.40621983198170386  p25=0.4026209888089655  p75=0.42031335702571787
tippy_tap_fraction median=0.05911154139156932
slip_ratio         median=0.021126113298601742
schedule_completion_rate=1.0
terminations={'schedule_complete': 10}

by hold:
       low: tripod median=0.48778157243677733 (n=10)
       mid: tripod median=0.4283929574729132 (n=10)
      high: tripod median=0.3236111410127992 (n=10)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
