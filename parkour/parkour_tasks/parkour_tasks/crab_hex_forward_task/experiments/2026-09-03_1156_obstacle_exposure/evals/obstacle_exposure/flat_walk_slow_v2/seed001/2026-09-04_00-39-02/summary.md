```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-03_18-16-56/model_24995.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5671872147009239  p25=0.5304479269377822  p75=0.596595180979064
tippy_tap_fraction median=0.22323232323232323
slip_ratio         median=0.2149131658820661
tracking_ratio     median=0.3865660049647014  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.93
terminations={'schedule_complete': 93, 'fall': 7}

by hold:
     stand: cmd 0.00 -> achieved 0.049 m/s | tripod median=0.6237497116515931 (n=100)
     creep: cmd 0.25 -> achieved 0.091 m/s | tripod median=0.5853591735461007 (n=100)
       low: cmd 0.35 -> achieved 0.144 m/s | tripod median=0.5252906856702931 (n=97)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
