```
=== crab-hex gait eval ===
scenario   : turn_walk_v1  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-31_10-46-19/model_19996.pt
episodes   : 100  unscored: 4
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5658736076619876  p25=0.5345209191267379  p75=0.5921398555516122
tippy_tap_fraction median=0.2478448275862069
slip_ratio         median=0.23479482830513254
tracking_ratio     median=0.4694655400047648  (achieved/commanded vx, walking holds; n=96)
schedule_completion_rate=0.86
terminations={'schedule_complete': 86, 'fall': 14}

by hold:
  straight: cmd 0.25 -> achieved 0.119 m/s | tripod median=0.4980922527438949 (n=96)
    turn_l: cmd 0.25 -> achieved 0.113 m/s | tripod median=0.5588741349091085 (n=87)
    turn_r: cmd 0.25 -> achieved 0.116 m/s | tripod median=0.6152337716529879 (n=87)
  turn_hard: cmd 0.25 -> achieved 0.118 m/s | tripod median=0.6279679705565149 (n=86)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
