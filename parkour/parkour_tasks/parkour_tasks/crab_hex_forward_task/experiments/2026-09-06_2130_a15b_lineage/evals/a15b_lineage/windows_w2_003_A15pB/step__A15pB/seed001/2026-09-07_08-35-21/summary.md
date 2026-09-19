```
=== crab-hex gait eval ===
scenario   : step__A15pB  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-07_02-19-49/model_14997.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.3975074345843129  p25=0.34153748850782817  p75=0.4626013378350821
tippy_tap_fraction median=0.26009145380006304
slip_ratio         median=0.2388539056162461
tracking_ratio     median=0.5475027562767228  (achieved/commanded vx, walking holds; n=87)
schedule_completion_rate=0.63
terminations={'fall': 37, 'schedule_complete': 63}

by hold:
     stand: cmd 0.00 -> achieved 0.076 m/s | tripod median=0.32559029827643643 (n=100)
     creep: cmd 0.25 -> achieved 0.152 m/s | tripod median=0.44288850483887654 (n=87)
       low: cmd 0.35 -> achieved 0.146 m/s | tripod median=0.4505448643168871 (n=66)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
