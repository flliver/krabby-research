```
=== crab-hex gait eval ===
scenario   : fwd__base  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-31_10-46-19/model_19996.pt
episodes   : 100  unscored: 15
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.3899866284777609  p25=0.3342439948035192  p75=0.42905164872424073
tippy_tap_fraction median=0.32939649578195973
slip_ratio         median=0.265199153508609
tracking_ratio     median=0.3618601090250449  (achieved/commanded vx, walking holds; n=85)
schedule_completion_rate=0.05
terminations={'fall': 95, 'schedule_complete': 5}

by hold:
       low: cmd 0.30 -> achieved 0.133 m/s | tripod median=0.44421403966702605 (n=85)
       mid: cmd 0.47 -> achieved 0.130 m/s | tripod median=0.34915296092987474 (n=63)
      high: cmd 0.65 -> achieved 0.171 m/s | tripod median=0.3097686539809153 (n=33)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
