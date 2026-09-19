```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-01_15-10-31/model_19996.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5271905111114641  p25=0.4711487794074709  p75=0.5658889126608165
tippy_tap_fraction median=0.25225675175730666
slip_ratio         median=0.23479452330467537
tracking_ratio     median=0.44067693813217823  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.6
terminations={'schedule_complete': 60, 'fall': 40}

by hold:
     stand: cmd 0.00 -> achieved 0.059 m/s | tripod median=0.5711146504653204 (n=100)
     creep: cmd 0.25 -> achieved 0.116 m/s | tripod median=0.5330394465940889 (n=100)
       low: cmd 0.35 -> achieved 0.139 m/s | tripod median=0.46230918660042297 (n=80)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
