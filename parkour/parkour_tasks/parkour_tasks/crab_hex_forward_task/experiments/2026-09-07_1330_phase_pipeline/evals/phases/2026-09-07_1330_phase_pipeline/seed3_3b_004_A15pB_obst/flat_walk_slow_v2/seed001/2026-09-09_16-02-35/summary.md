```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Student-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_student/2026-09-09_02-06-51/model_29994.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.4811691568731157  p25=0.4159543818505281  p75=0.5330042290073946
tippy_tap_fraction median=0.24336401227157528
slip_ratio         median=0.1914223428787381
tracking_ratio     median=0.6019509786070906  (achieved/commanded vx, walking holds; n=94)
schedule_completion_rate=0.62
terminations={'schedule_complete': 62, 'fall': 38}

by hold:
     stand: cmd 0.00 -> achieved 0.111 m/s | tripod median=0.44492424575783507 (n=100)
     creep: cmd 0.25 -> achieved 0.162 m/s | tripod median=0.5317568440502489 (n=94)
       low: cmd 0.35 -> achieved 0.181 m/s | tripod median=0.5023776390971766 (n=76)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
