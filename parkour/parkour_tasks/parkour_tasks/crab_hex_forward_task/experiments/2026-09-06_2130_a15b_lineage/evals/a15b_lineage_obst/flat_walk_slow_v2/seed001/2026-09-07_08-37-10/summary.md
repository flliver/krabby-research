```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-07_02-19-49/model_14997.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.38806897239253724  p25=0.3257625624659086  p75=0.47806239986111626
tippy_tap_fraction median=0.24117158288325719
slip_ratio         median=0.2077735752516638
tracking_ratio     median=0.5758387733657385  (achieved/commanded vx, walking holds; n=82)
schedule_completion_rate=0.58
terminations={'schedule_complete': 58, 'fall': 42}

by hold:
     stand: cmd 0.00 -> achieved 0.075 m/s | tripod median=0.3303053334213346 (n=100)
     creep: cmd 0.25 -> achieved 0.158 m/s | tripod median=0.46848762116697984 (n=81)
       low: cmd 0.35 -> achieved 0.175 m/s | tripod median=0.5060769902728446 (n=64)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
