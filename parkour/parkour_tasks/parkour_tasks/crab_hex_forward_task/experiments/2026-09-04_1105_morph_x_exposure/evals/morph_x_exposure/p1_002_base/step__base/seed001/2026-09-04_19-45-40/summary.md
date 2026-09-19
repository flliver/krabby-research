```
=== crab-hex gait eval ===
scenario   : step__base  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-04_13-18-07/model_4999.pt
episodes   : 100  unscored: 1
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.43496958739541025  p25=0.36954587213541057  p75=0.5044695987782462
tippy_tap_fraction median=0.25987012987012986
slip_ratio         median=0.269116181908384
tracking_ratio     median=0.5802186591009395  (achieved/commanded vx, walking holds; n=93)
schedule_completion_rate=0.29
terminations={'fall': 71, 'schedule_complete': 29}

by hold:
     stand: cmd 0.00 -> achieved 0.058 m/s | tripod median=0.40653231615661595 (n=99)
     creep: cmd 0.25 -> achieved 0.165 m/s | tripod median=0.43966909339803156 (n=92)
       low: cmd 0.35 -> achieved 0.155 m/s | tripod median=0.5016875789764956 (n=38)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
