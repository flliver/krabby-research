```
=== crab-hex gait eval ===
scenario   : slow__A20pB  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-05_07-56-31/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5622011742573993  p25=0.5180095192976012  p75=0.5941812123904013
tippy_tap_fraction median=0.31550198627663417
slip_ratio         median=0.2547576711665013
tracking_ratio     median=0.6860686664922129  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.98
terminations={'schedule_complete': 98, 'fall': 2}

by hold:
     stand: cmd 0.00 -> achieved 0.122 m/s | tripod median=0.3694978683785616 (n=100)
     creep: cmd 0.25 -> achieved 0.227 m/s | tripod median=0.705088005993002 (n=100)
       low: cmd 0.35 -> achieved 0.162 m/s | tripod median=0.6143569863589073 (n=100)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
