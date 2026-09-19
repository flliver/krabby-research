```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-05_03-00-33/model_4999.pt
episodes   : 100  unscored: 2
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.3758611759587116  p25=0.31549423062069176  p75=0.4180020912817075
tippy_tap_fraction median=0.31567944250871083
slip_ratio         median=0.2378156938268265
tracking_ratio     median=0.5760226827925108  (achieved/commanded vx, walking holds; n=61)
schedule_completion_rate=0.29
terminations={'fall': 71, 'schedule_complete': 29}

by hold:
     stand: cmd 0.00 -> achieved 0.047 m/s | tripod median=0.35738859303885784 (n=98)
     creep: cmd 0.25 -> achieved 0.180 m/s | tripod median=0.48682795021219016 (n=61)
       low: cmd 0.35 -> achieved 0.143 m/s | tripod median=0.30873967775145167 (n=47)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
