```
=== crab-hex gait eval ===
scenario   : step__B  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-31_10-46-19/model_19996.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5827857517206368  p25=0.5346087080360544  p75=0.64490100978049
tippy_tap_fraction median=0.23563218390804597
slip_ratio         median=0.20749515349974468
tracking_ratio     median=0.4378722547285061  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.83
terminations={'schedule_complete': 83, 'fall': 17}

by hold:
     stand: cmd 0.00 -> achieved 0.069 m/s | tripod median=0.6763148287777865 (n=100)
     creep: cmd 0.25 -> achieved 0.119 m/s | tripod median=0.5591102681815197 (n=100)
       low: cmd 0.35 -> achieved 0.136 m/s | tripod median=0.5413329892956467 (n=90)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
