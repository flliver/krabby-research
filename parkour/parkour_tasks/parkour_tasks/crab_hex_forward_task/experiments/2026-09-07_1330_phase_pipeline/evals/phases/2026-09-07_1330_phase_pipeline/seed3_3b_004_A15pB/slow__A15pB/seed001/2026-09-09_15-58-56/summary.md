```
=== crab-hex gait eval ===
scenario   : slow__A15pB  (Isaac-Crab-Hex-Student-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_student/2026-09-09_02-06-51/model_29994.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5067234441866605  p25=0.4505473159461586  p75=0.5425354607284527
tippy_tap_fraction median=0.25817133091219824
slip_ratio         median=0.19988100714874646
tracking_ratio     median=0.6486240178746617  (achieved/commanded vx, walking holds; n=98)
schedule_completion_rate=0.81
terminations={'schedule_complete': 81, 'fall': 19}

by hold:
     stand: cmd 0.00 -> achieved 0.128 m/s | tripod median=0.5289720035519545 (n=100)
     creep: cmd 0.25 -> achieved 0.174 m/s | tripod median=0.5256484141664499 (n=98)
       low: cmd 0.35 -> achieved 0.192 m/s | tripod median=0.4702610634774502 (n=86)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
