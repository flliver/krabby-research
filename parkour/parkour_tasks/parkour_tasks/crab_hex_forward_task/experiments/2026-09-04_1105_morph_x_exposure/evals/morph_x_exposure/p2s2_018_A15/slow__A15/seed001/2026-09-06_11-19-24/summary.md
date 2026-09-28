```
=== crab-hex gait eval ===
scenario   : slow__A15  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-06_05-05-22/model_9998.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5399692911059966  p25=0.5021244005778155  p75=0.5773458858089926
tippy_tap_fraction median=0.2241901776384535
slip_ratio         median=0.19540664842557637
tracking_ratio     median=0.5837363320001672  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}

by hold:
     stand: cmd 0.00 -> achieved 0.066 m/s | tripod median=0.4192878181011955 (n=100)
     creep: cmd 0.25 -> achieved 0.166 m/s | tripod median=0.6226093421716876 (n=100)
       low: cmd 0.35 -> achieved 0.175 m/s | tripod median=0.6041443770621632 (n=100)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
