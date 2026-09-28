```
=== crab-hex gait eval ===
scenario   : slow__A20  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-03_07-16-23/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.3185568617644328  p25=0.28221742251713333  p75=0.3545322463872412
tippy_tap_fraction median=0.39952904238618525
slip_ratio         median=0.3902701394266162
tracking_ratio     median=0.7065984543064272  (achieved/commanded vx, walking holds; n=99)
schedule_completion_rate=0.76
terminations={'schedule_complete': 76, 'fall': 24}

by hold:
     stand: cmd 0.00 -> achieved 0.008 m/s | tripod median=0.03574332172175032 (n=100)
     creep: cmd 0.25 -> achieved 0.222 m/s | tripod median=0.5195502928049698 (n=97)
       low: cmd 0.35 -> achieved 0.185 m/s | tripod median=0.40798497700398106 (n=84)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
