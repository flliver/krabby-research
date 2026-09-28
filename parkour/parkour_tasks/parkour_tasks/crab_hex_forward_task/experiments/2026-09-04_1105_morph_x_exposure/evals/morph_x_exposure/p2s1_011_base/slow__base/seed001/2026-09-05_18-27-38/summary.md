```
=== crab-hex gait eval ===
scenario   : slow__base  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-05_12-03-33/model_4999.pt
episodes   : 100  unscored: 2
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.41830376014443216  p25=0.28885236440493867  p75=0.4776062226225519
tippy_tap_fraction median=0.3474115598590054
slip_ratio         median=0.26779683447189406
tracking_ratio     median=0.6959143214668648  (achieved/commanded vx, walking holds; n=69)
schedule_completion_rate=0.28
terminations={'fall': 72, 'schedule_complete': 28}

by hold:
     stand: cmd 0.00 -> achieved 0.072 m/s | tripod median=0.33846517951746896 (n=98)
     creep: cmd 0.25 -> achieved 0.195 m/s | tripod median=0.4750640320980555 (n=67)
       low: cmd 0.35 -> achieved 0.204 m/s | tripod median=0.5740318656581405 (n=32)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
