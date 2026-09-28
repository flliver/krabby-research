```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-04_13-18-07/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.4380459410080435  p25=0.39051210461622726  p75=0.50481339451875
tippy_tap_fraction median=0.24820382165605095
slip_ratio         median=0.26456858359812985
tracking_ratio     median=0.5500469838460462  (achieved/commanded vx, walking holds; n=96)
schedule_completion_rate=0.39
terminations={'schedule_complete': 39, 'fall': 61}

by hold:
     stand: cmd 0.00 -> achieved 0.055 m/s | tripod median=0.39690559609039744 (n=100)
     creep: cmd 0.25 -> achieved 0.154 m/s | tripod median=0.5018448619475102 (n=96)
       low: cmd 0.35 -> achieved 0.154 m/s | tripod median=0.45999309137413813 (n=61)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
