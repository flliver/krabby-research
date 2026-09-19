```
=== crab-hex gait eval ===
scenario   : step__base  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-05_14-32-24/model_9998.pt
episodes   : 100  unscored: 1
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.4680396432793443  p25=0.42142565260359693  p75=0.5036740890417706
tippy_tap_fraction median=0.32964481698124587
slip_ratio         median=0.24308804056404973
tracking_ratio     median=0.6440573808903921  (achieved/commanded vx, walking holds; n=97)
schedule_completion_rate=0.33
terminations={'fall': 67, 'schedule_complete': 33}

by hold:
     stand: cmd 0.00 -> achieved 0.066 m/s | tripod median=0.464233004088404 (n=99)
     creep: cmd 0.25 -> achieved 0.180 m/s | tripod median=0.47153078215653355 (n=92)
       low: cmd 0.35 -> achieved 0.155 m/s | tripod median=0.47027593531860834 (n=40)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
