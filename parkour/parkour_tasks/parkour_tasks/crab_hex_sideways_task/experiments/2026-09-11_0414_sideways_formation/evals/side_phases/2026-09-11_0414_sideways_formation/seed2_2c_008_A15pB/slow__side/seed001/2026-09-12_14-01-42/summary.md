```
=== crab-hex gait eval ===
scenario   : slow__side  (Isaac-Crab-Hex-Side-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_side_teacher/2026-09-12_07-45-40/model_19996.pt
episodes   : 100  unscored: 1
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.45589496412602387  p25=0.4165814025037503  p75=0.4812352836271473
tippy_tap_fraction median=0.35311355311355314
slip_ratio         median=0.3931280123005546
tracking_ratio     median=0.7263765830457827  (achieved/commanded vx, walking holds; n=98)
schedule_completion_rate=0.9
terminations={'schedule_complete': 90, 'fall': 10}

by hold:
     stand: cmd 0.00 -> achieved 0.005 m/s | tripod median=0.0 (n=91)
     creep: cmd 0.25 -> achieved 0.208 m/s | tripod median=0.6624868702902454 (n=98)
       low: cmd 0.35 -> achieved 0.220 m/s | tripod median=0.6856092024205369 (n=91)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
