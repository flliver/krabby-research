```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-31_23-49-32/model_9998.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6140118654178054  p25=0.5906235180419918  p75=0.6430806611222305
tippy_tap_fraction median=0.24454036581520025
slip_ratio         median=0.23815705151797772
tracking_ratio     median=0.4796515871158626  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.93
terminations={'schedule_complete': 93, 'fall': 7}

by hold:
     stand: cmd 0.00 -> achieved 0.052 m/s | tripod median=0.6388150958384282 (n=100)
     creep: cmd 0.25 -> achieved 0.134 m/s | tripod median=0.6837218795548345 (n=100)
       low: cmd 0.35 -> achieved 0.147 m/s | tripod median=0.5448071087696535 (n=100)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
