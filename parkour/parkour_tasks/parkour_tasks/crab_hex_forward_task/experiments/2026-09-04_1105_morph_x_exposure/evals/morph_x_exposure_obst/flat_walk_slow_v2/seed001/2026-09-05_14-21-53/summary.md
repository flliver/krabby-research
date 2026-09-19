```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-05_07-56-31/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.45854533754512505  p25=0.39341711309066835  p75=0.5209778888128005
tippy_tap_fraction median=0.3402014652014652
slip_ratio         median=0.28294853893474803
tracking_ratio     median=0.6463788854936664  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.51
terminations={'schedule_complete': 51, 'fall': 49}

by hold:
     stand: cmd 0.00 -> achieved 0.092 m/s | tripod median=0.32549025099510126 (n=100)
     creep: cmd 0.25 -> achieved 0.210 m/s | tripod median=0.6051946316621062 (n=100)
       low: cmd 0.35 -> achieved 0.151 m/s | tripod median=0.5179630682483395 (n=74)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
