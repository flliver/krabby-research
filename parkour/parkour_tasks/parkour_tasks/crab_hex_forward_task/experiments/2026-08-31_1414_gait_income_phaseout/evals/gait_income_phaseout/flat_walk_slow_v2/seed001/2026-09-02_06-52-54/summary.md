```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-02_00-42-53/model_29994.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.562982041877398  p25=0.5284825354879638  p75=0.6037678378633944
tippy_tap_fraction median=0.2529477042585657
slip_ratio         median=0.23445821507449094
tracking_ratio     median=0.4067250587464134  (achieved/commanded vx, walking holds; n=99)
schedule_completion_rate=0.79
terminations={'schedule_complete': 79, 'fall': 21}

by hold:
     stand: cmd 0.00 -> achieved 0.064 m/s | tripod median=0.7199781462968337 (n=100)
     creep: cmd 0.25 -> achieved 0.112 m/s | tripod median=0.4800069284598927 (n=99)
       low: cmd 0.35 -> achieved 0.125 m/s | tripod median=0.4712952864587098 (n=84)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
