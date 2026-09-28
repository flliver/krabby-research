```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-30_03-51-02/model_9998.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.3760248527310607  p25=0.32619020592357706  p75=0.416227711058035
tippy_tap_fraction median=0.27500510308226167
slip_ratio         median=0.27268128829325755
tracking_ratio     median=0.44012126240359195  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.77
terminations={'schedule_complete': 77, 'fall': 23}

by hold:
     stand: cmd 0.00 -> achieved 0.000 m/s | tripod median=0.046789442504056124 (n=100)
     creep: cmd 0.25 -> achieved 0.122 m/s | tripod median=0.5591173843127992 (n=100)
       low: cmd 0.35 -> achieved 0.135 m/s | tripod median=0.5764969796634827 (n=94)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
