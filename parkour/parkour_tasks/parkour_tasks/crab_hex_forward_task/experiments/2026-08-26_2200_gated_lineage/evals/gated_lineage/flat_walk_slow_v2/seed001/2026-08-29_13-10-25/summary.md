```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-29_07-50-25/model_9997.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.40343904034038025  p25=0.371958828517604  p75=0.43002648338918004
tippy_tap_fraction median=0.2895058494848961
slip_ratio         median=0.29608004723277737
tracking_ratio     median=0.5045055253489152  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.98
terminations={'schedule_complete': 98, 'fall': 2}

by hold:
     stand: cmd 0.00 -> achieved 0.009 m/s | tripod median=0.08408643887670172 (n=100)
     creep: cmd 0.25 -> achieved 0.148 m/s | tripod median=0.569243754548622 (n=100)
       low: cmd 0.35 -> achieved 0.143 m/s | tripod median=0.5413269769488822 (n=98)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
