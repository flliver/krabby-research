```
=== crab-hex gait eval ===
scenario   : step__A20  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-05_00-31-35/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.4571127479368795  p25=0.39558982426572076  p75=0.5022686949816719
tippy_tap_fraction median=0.2288690476190476
slip_ratio         median=0.2088920209586927
tracking_ratio     median=0.6056639613472647  (achieved/commanded vx, walking holds; n=70)
schedule_completion_rate=0.33
terminations={'fall': 67, 'schedule_complete': 33}

by hold:
     stand: cmd 0.00 -> achieved 0.028 m/s | tripod median=0.39547924452214644 (n=100)
     creep: cmd 0.25 -> achieved 0.175 m/s | tripod median=0.5379097118435001 (n=69)
       low: cmd 0.35 -> achieved 0.162 m/s | tripod median=0.4853495266420387 (n=52)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
