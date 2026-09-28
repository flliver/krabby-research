```
=== crab-hex gait eval ===
scenario   : slow__base  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-05_14-32-24/model_9998.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5203897236254978  p25=0.48385804080381567  p75=0.5597596563490591
tippy_tap_fraction median=0.32278755164600825
slip_ratio         median=0.24891151664310482
tracking_ratio     median=0.6252834987389762  (achieved/commanded vx, walking holds; n=98)
schedule_completion_rate=0.68
terminations={'schedule_complete': 68, 'fall': 32}

by hold:
     stand: cmd 0.00 -> achieved 0.076 m/s | tripod median=0.47166208506805285 (n=100)
     creep: cmd 0.25 -> achieved 0.178 m/s | tripod median=0.5312351070197453 (n=98)
       low: cmd 0.35 -> achieved 0.182 m/s | tripod median=0.5797577753780866 (n=78)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
