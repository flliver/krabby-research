```
=== crab-hex gait eval ===
scenario   : step__A15pB  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-07_06-57-52/model_24995.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.3444787145148217  p25=0.2646163891861097  p75=0.428654995537174
tippy_tap_fraction median=0.25
slip_ratio         median=0.1852319466872617
tracking_ratio     median=0.5074021124993081  (achieved/commanded vx, walking holds; n=73)
schedule_completion_rate=0.42
terminations={'fall': 58, 'schedule_complete': 42}

by hold:
     stand: cmd 0.00 -> achieved 0.105 m/s | tripod median=0.30271743079794633 (n=100)
     creep: cmd 0.25 -> achieved 0.151 m/s | tripod median=0.3884017951509365 (n=71)
       low: cmd 0.35 -> achieved 0.134 m/s | tripod median=0.3844788262868171 (n=48)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
