```
=== crab-hex gait eval ===
scenario   : slow__side  (Isaac-Crab-Hex-Side-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_side_teacher/2026-09-12_05-24-07/model_14800.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.45190871016878986  p25=0.39857930758594995  p75=0.4908824244205049
tippy_tap_fraction median=0.3673769923769924
slip_ratio         median=0.3832971868830691
tracking_ratio     median=0.7375491817210033  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.93
terminations={'schedule_complete': 93, 'fall': 7}

by hold:
     stand: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.0 (n=86)
     creep: cmd 0.25 -> achieved 0.210 m/s | tripod median=0.6091782874443779 (n=100)
       low: cmd 0.35 -> achieved 0.224 m/s | tripod median=0.6787147902696364 (n=94)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
