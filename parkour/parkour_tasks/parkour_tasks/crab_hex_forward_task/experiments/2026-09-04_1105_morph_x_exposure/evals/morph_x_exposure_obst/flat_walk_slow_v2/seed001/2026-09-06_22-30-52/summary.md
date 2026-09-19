```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-06_16-03-30/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.40509727949193575  p25=0.3404592708078356  p75=0.46603619106834987
tippy_tap_fraction median=0.27526505203773954
slip_ratio         median=0.26562335783653795
tracking_ratio     median=0.6677155411625336  (achieved/commanded vx, walking holds; n=82)
schedule_completion_rate=0.33
terminations={'schedule_complete': 33, 'fall': 67}

by hold:
     stand: cmd 0.00 -> achieved 0.031 m/s | tripod median=0.3629733065749696 (n=100)
     creep: cmd 0.25 -> achieved 0.182 m/s | tripod median=0.42122557512720954 (n=81)
       low: cmd 0.35 -> achieved 0.205 m/s | tripod median=0.4775443932299295 (n=59)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
