```
=== crab-hex gait eval ===
scenario   : step__A15pB  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-07_09-16-39/model_29994.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.3500800431428062  p25=0.29716069691584357  p75=0.4290511896934509
tippy_tap_fraction median=0.24675105485232068
slip_ratio         median=0.21212233224683189
tracking_ratio     median=0.5852832582697812  (achieved/commanded vx, walking holds; n=94)
schedule_completion_rate=0.6
terminations={'fall': 40, 'schedule_complete': 60}

by hold:
     stand: cmd 0.00 -> achieved 0.118 m/s | tripod median=0.2581823521212751 (n=100)
     creep: cmd 0.25 -> achieved 0.168 m/s | tripod median=0.42224947785736316 (n=94)
       low: cmd 0.35 -> achieved 0.168 m/s | tripod median=0.49713377048173846 (n=66)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
