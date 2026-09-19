```
=== crab-hex gait eval ===
scenario   : step__A10pB  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-31_10-46-19/model_19996.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6092168015030159  p25=0.5499801239977599  p75=0.6717105395238497
tippy_tap_fraction median=0.23546386688770185
slip_ratio         median=0.21733005876779332
tracking_ratio     median=0.3917149412083681  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.99
terminations={'schedule_complete': 99, 'fall': 1}

by hold:
     stand: cmd 0.00 -> achieved 0.067 m/s | tripod median=0.6642720069598761 (n=100)
     creep: cmd 0.25 -> achieved 0.107 m/s | tripod median=0.6205124284822444 (n=100)
       low: cmd 0.35 -> achieved 0.123 m/s | tripod median=0.6169761920152258 (n=99)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
