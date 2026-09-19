```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-05_14-32-24/model_9998.pt
episodes   : 100  unscored: 1
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.48424635723126747  p25=0.43842933958570873  p75=0.513478935924429
tippy_tap_fraction median=0.34444753172177456
slip_ratio         median=0.2553911998154792
tracking_ratio     median=0.6594308012496266  (achieved/commanded vx, walking holds; n=98)
schedule_completion_rate=0.31
terminations={'fall': 69, 'schedule_complete': 31}

by hold:
     stand: cmd 0.00 -> achieved 0.064 m/s | tripod median=0.45881475724905946 (n=99)
     creep: cmd 0.25 -> achieved 0.174 m/s | tripod median=0.5113323007068767 (n=97)
       low: cmd 0.35 -> achieved 0.171 m/s | tripod median=0.5179801829961042 (n=48)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
