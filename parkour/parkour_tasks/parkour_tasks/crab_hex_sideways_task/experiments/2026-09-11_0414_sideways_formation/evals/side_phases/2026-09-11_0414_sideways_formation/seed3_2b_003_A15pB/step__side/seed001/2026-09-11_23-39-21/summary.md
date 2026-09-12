```
=== crab-hex gait eval ===
scenario   : step__side  (Isaac-Crab-Hex-Side-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_side_teacher/2026-09-11_17-19-40/model_14997.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.2957851794375196  p25=0.2038357325094919  p75=0.36709587365722396
tippy_tap_fraction median=0.4039750957854406
slip_ratio         median=0.4520315947463116
tracking_ratio     median=0.5227164226409088  (achieved/commanded vx, walking holds; n=98)
schedule_completion_rate=0.82
terminations={'fall': 18, 'schedule_complete': 82}

by hold:
     stand: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.0 (n=82)
     creep: cmd 0.25 -> achieved 0.177 m/s | tripod median=0.5193305049606688 (n=98)
       low: cmd 0.35 -> achieved 0.118 m/s | tripod median=0.3334540391324972 (n=86)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
