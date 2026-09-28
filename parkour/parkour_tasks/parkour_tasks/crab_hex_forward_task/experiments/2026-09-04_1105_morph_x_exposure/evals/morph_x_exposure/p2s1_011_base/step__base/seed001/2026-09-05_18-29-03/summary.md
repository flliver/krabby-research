```
=== crab-hex gait eval ===
scenario   : step__base  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-05_12-03-33/model_4999.pt
episodes   : 100  unscored: 2
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.40312293692638657  p25=0.32953770996116577  p75=0.4533908004334903
tippy_tap_fraction median=0.3579607415485278
slip_ratio         median=0.2679968345432444
tracking_ratio     median=0.6799819727987051  (achieved/commanded vx, walking holds; n=65)
schedule_completion_rate=0.09
terminations={'fall': 91, 'schedule_complete': 9}

by hold:
     stand: cmd 0.00 -> achieved 0.075 m/s | tripod median=0.38569897899319905 (n=98)
     creep: cmd 0.25 -> achieved 0.175 m/s | tripod median=0.4260688863856621 (n=59)
       low: cmd 0.35 -> achieved 0.180 m/s | tripod median=0.5097103541649208 (n=17)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
