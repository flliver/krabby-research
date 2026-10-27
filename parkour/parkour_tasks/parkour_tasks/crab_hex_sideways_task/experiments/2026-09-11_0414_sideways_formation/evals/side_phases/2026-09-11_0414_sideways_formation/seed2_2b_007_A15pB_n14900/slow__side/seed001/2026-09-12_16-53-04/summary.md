```
=== crab-hex gait eval ===
scenario   : slow__side  (Isaac-Crab-Hex-Side-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_side_teacher/2026-09-12_05-24-07/model_14900.pt
episodes   : 100  unscored: 1
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.43172529484703587  p25=0.38950894429052274  p75=0.45987922906853324
tippy_tap_fraction median=0.35985611510791365
slip_ratio         median=0.4395983857943361
tracking_ratio     median=0.7340755505285813  (achieved/commanded vx, walking holds; n=98)
schedule_completion_rate=0.87
terminations={'schedule_complete': 87, 'fall': 13}

by hold:
     stand: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.0 (n=90)
     creep: cmd 0.25 -> achieved 0.214 m/s | tripod median=0.6336184701318508 (n=98)
       low: cmd 0.35 -> achieved 0.216 m/s | tripod median=0.6551605778562248 (n=90)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
