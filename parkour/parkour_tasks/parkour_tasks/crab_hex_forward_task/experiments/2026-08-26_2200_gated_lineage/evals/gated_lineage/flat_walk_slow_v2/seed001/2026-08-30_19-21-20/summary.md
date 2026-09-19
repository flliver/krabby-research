```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-30_13-39-21/model_14996.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.40846064141613425  p25=0.3802141396282024  p75=0.4380466853472938
tippy_tap_fraction median=0.2812904745084187
slip_ratio         median=0.2969733664905485
tracking_ratio     median=0.5214773479779429  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.99
terminations={'schedule_complete': 99, 'fall': 1}

by hold:
     stand: cmd 0.00 -> achieved 0.016 m/s | tripod median=0.17242968431067554 (n=100)
     creep: cmd 0.25 -> achieved 0.145 m/s | tripod median=0.5737309552000701 (n=100)
       low: cmd 0.35 -> achieved 0.160 m/s | tripod median=0.4791900107294087 (n=99)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
