```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-31_17-46-45/model_9997.pt
episodes   : 100  unscored: 1
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5735918439770997  p25=0.5327388734522689  p75=0.6148660041502219
tippy_tap_fraction median=0.2390096618357488
slip_ratio         median=0.20407588480839256
tracking_ratio     median=0.4386150033297596  (achieved/commanded vx, walking holds; n=99)
schedule_completion_rate=0.99
terminations={'schedule_complete': 99, 'fall': 1}

by hold:
     stand: cmd 0.00 -> achieved 0.034 m/s | tripod median=0.4731185346451433 (n=99)
     creep: cmd 0.25 -> achieved 0.112 m/s | tripod median=0.6373544561368673 (n=99)
       low: cmd 0.35 -> achieved 0.149 m/s | tripod median=0.614019892709808 (n=99)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
