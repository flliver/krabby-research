```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-06_02-36-19/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.4464407998442182  p25=0.3756718690006466  p75=0.5216733355199905
tippy_tap_fraction median=0.2326312710253035
slip_ratio         median=0.2508231596310925
tracking_ratio     median=0.5512439228764068  (achieved/commanded vx, walking holds; n=70)
schedule_completion_rate=0.48
terminations={'fall': 52, 'schedule_complete': 48}

by hold:
     stand: cmd 0.00 -> achieved 0.014 m/s | tripod median=0.3923576406035616 (n=100)
     creep: cmd 0.25 -> achieved 0.152 m/s | tripod median=0.5673571287095094 (n=70)
       low: cmd 0.35 -> achieved 0.182 m/s | tripod median=0.514892535288666 (n=59)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
