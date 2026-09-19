```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-01_13-17-11/model_16996.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.556079812805568  p25=0.5104489292416583  p75=0.5967425860173818
tippy_tap_fraction median=0.24424603174603174
slip_ratio         median=0.20267977905727053
tracking_ratio     median=0.44325705219195877  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.98
terminations={'schedule_complete': 98, 'fall': 2}

by hold:
     stand: cmd 0.00 -> achieved 0.055 m/s | tripod median=0.5874226333664347 (n=100)
     creep: cmd 0.25 -> achieved 0.113 m/s | tripod median=0.5860928662059733 (n=100)
       low: cmd 0.35 -> achieved 0.155 m/s | tripod median=0.519499244593575 (n=99)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
