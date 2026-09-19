```
=== crab-hex gait eval ===
scenario   : step__A15  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-06_02-36-19/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.4493816246572469  p25=0.31817230882194114  p75=0.5238129366305948
tippy_tap_fraction median=0.2380448318804483
slip_ratio         median=0.24697439004601485
tracking_ratio     median=0.5318218992709453  (achieved/commanded vx, walking holds; n=68)
schedule_completion_rate=0.45
terminations={'fall': 55, 'schedule_complete': 45}

by hold:
     stand: cmd 0.00 -> achieved 0.015 m/s | tripod median=0.39985333533910605 (n=100)
     creep: cmd 0.25 -> achieved 0.153 m/s | tripod median=0.5425921177581554 (n=68)
       low: cmd 0.35 -> achieved 0.159 m/s | tripod median=0.47891083804616197 (n=54)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
