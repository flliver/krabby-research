```
=== crab-hex gait eval ===
scenario   : step__A10  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-03_02-14-33/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.1758564896369334  p25=0.0937105722798783  p75=0.2550245336355058
tippy_tap_fraction median=0.4435336976320583
slip_ratio         median=0.5822006047870867
tracking_ratio     median=0.699955321661879  (achieved/commanded vx, walking holds; n=92)
schedule_completion_rate=0.1
terminations={'fall': 90, 'schedule_complete': 10}

by hold:
     stand: cmd 0.00 -> achieved -0.001 m/s | tripod median=0.0 (n=92)
     creep: cmd 0.25 -> achieved 0.184 m/s | tripod median=0.3569790822877572 (n=92)
       low: cmd 0.35 -> achieved 0.149 m/s | tripod median=0.324232085975362 (n=18)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
