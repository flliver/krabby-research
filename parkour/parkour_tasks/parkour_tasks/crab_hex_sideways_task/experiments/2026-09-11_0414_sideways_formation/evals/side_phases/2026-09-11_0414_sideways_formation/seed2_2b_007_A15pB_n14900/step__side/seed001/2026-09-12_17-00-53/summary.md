```
=== crab-hex gait eval ===
scenario   : step__side  (Isaac-Crab-Hex-Side-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_side_teacher/2026-09-12_05-24-07/model_14900.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.3302135133522959  p25=0.24002616584214304  p75=0.3925507302099124
tippy_tap_fraction median=0.36099812265331666
slip_ratio         median=0.46695122498557834
tracking_ratio     median=0.5761672414118477  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.75
terminations={'fall': 25, 'schedule_complete': 75}

by hold:
     stand: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.0 (n=86)
     creep: cmd 0.25 -> achieved 0.188 m/s | tripod median=0.5636144252358062 (n=100)
       low: cmd 0.35 -> achieved 0.135 m/s | tripod median=0.40615141731461657 (n=84)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
