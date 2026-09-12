```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_side  (Isaac-Crab-Hex-Side-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_side_teacher/2026-09-11_19-43-19/model_19996.pt
episodes   : 100  unscored: 2
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.3551538744348295  p25=0.30693444318736574  p75=0.3891443713534487
tippy_tap_fraction median=0.399334362338414
slip_ratio         median=0.3956106465597743
tracking_ratio     median=0.6107038972731685  (achieved/commanded vx, walking holds; n=94)
schedule_completion_rate=0.82
terminations={'schedule_complete': 82, 'fall': 18}

by hold:
     stand: cmd 0.00 -> achieved 0.005 m/s | tripod median=0.0 (n=87)
     creep: cmd 0.25 -> achieved 0.191 m/s | tripod median=0.5651967569397082 (n=94)
       low: cmd 0.35 -> achieved 0.162 m/s | tripod median=0.4444130469105466 (n=85)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
