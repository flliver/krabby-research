```
=== crab-hex gait eval ===
scenario   : slow__A15  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-06_02-36-19/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.4940783879045345  p25=0.4462892811728811  p75=0.545230454533155
tippy_tap_fraction median=0.23202307199316385
slip_ratio         median=0.23403578141632125
tracking_ratio     median=0.5951211479563178  (achieved/commanded vx, walking holds; n=91)
schedule_completion_rate=0.78
terminations={'fall': 22, 'schedule_complete': 78}

by hold:
     stand: cmd 0.00 -> achieved 0.036 m/s | tripod median=0.37961872473506 (n=100)
     creep: cmd 0.25 -> achieved 0.165 m/s | tripod median=0.5595170219092186 (n=91)
       low: cmd 0.35 -> achieved 0.186 m/s | tripod median=0.5861321863255093 (n=87)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
