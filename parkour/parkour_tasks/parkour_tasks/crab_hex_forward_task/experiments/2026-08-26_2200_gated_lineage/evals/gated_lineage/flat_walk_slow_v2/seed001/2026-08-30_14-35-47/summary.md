```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-30_09-07-47/model_14996.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.41775044872646433  p25=0.39355036248571734  p75=0.4351696524810005
tippy_tap_fraction median=0.28887361526643485
slip_ratio         median=0.31925688370090477
tracking_ratio     median=0.47003563723158504  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.99
terminations={'schedule_complete': 99, 'fall': 1}

by hold:
     stand: cmd 0.00 -> achieved 0.006 m/s | tripod median=0.0948998592514016 (n=100)
     creep: cmd 0.25 -> achieved 0.127 m/s | tripod median=0.5704490646496216 (n=100)
       low: cmd 0.35 -> achieved 0.149 m/s | tripod median=0.5812659398897394 (n=99)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
