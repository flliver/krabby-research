```
=== crab-hex gait eval ===
scenario   : slow__side  (Isaac-Crab-Hex-Side-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_side_teacher/2026-09-11_17-19-40/model_14997.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.3784674805086814  p25=0.3404517094881454  p75=0.4351624801554726
tippy_tap_fraction median=0.4132185301964589
slip_ratio         median=0.38086029404605126
tracking_ratio     median=0.6628972641993838  (achieved/commanded vx, walking holds; n=97)
schedule_completion_rate=0.91
terminations={'schedule_complete': 91, 'fall': 9}

by hold:
     stand: cmd 0.00 -> achieved 0.006 m/s | tripod median=0.0013123592317104379 (n=87)
     creep: cmd 0.25 -> achieved 0.198 m/s | tripod median=0.5821002206894949 (n=97)
       low: cmd 0.35 -> achieved 0.189 m/s | tripod median=0.5103813132774022 (n=93)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
