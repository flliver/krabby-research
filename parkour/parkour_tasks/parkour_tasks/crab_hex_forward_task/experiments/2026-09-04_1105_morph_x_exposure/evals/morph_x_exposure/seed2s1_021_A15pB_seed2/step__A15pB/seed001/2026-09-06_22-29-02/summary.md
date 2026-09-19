```
=== crab-hex gait eval ===
scenario   : step__A15pB  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-06_16-03-30/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.41979180774114666  p25=0.34622385105558917  p75=0.46442890508348617
tippy_tap_fraction median=0.2716049382716049
slip_ratio         median=0.28638786652694037
tracking_ratio     median=0.6542569373507653  (achieved/commanded vx, walking holds; n=85)
schedule_completion_rate=0.48
terminations={'fall': 52, 'schedule_complete': 48}

by hold:
     stand: cmd 0.00 -> achieved 0.028 m/s | tripod median=0.3619358256277333 (n=100)
     creep: cmd 0.25 -> achieved 0.194 m/s | tripod median=0.44347251253483416 (n=85)
       low: cmd 0.35 -> achieved 0.185 m/s | tripod median=0.4550244692007213 (n=70)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
