```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-31_10-46-19/model_19996.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.605633372886274  p25=0.5639903840652485  p75=0.6342258905860918
tippy_tap_fraction median=0.23449488491048592
slip_ratio         median=0.19913826383456076
tracking_ratio     median=0.43340404589559645  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.99
terminations={'schedule_complete': 99, 'fall': 1}

by hold:
     stand: cmd 0.00 -> achieved 0.066 m/s | tripod median=0.6668327141978829 (n=100)
     creep: cmd 0.25 -> achieved 0.116 m/s | tripod median=0.5860007072718516 (n=100)
       low: cmd 0.35 -> achieved 0.142 m/s | tripod median=0.583794712852842 (n=100)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
