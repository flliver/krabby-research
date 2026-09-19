```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-02_09-51-23/model_29994.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5959995052878146  p25=0.5496310297874778  p75=0.6297015930979046
tippy_tap_fraction median=0.21964620745108548
slip_ratio         median=0.18009247798939204
tracking_ratio     median=0.751681970698141  (achieved/commanded vx, walking holds; n=92)
schedule_completion_rate=0.19
terminations={'fall': 81, 'schedule_complete': 19}

by hold:
     stand: cmd 0.00 -> achieved 0.089 m/s | tripod median=0.6208784033349516 (n=100)
     creep: cmd 0.25 -> achieved 0.186 m/s | tripod median=0.5473590253064717 (n=90)
       low: cmd 0.35 -> achieved 0.151 m/s | tripod median=0.5416123520812449 (n=31)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
