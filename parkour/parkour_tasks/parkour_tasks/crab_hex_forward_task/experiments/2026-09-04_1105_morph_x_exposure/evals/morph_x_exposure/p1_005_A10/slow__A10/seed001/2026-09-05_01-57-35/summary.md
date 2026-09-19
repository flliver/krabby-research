```
=== crab-hex gait eval ===
scenario   : slow__A10  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-04_19-33-30/model_4999.pt
episodes   : 100  unscored: 3
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.36959458826577013  p25=0.32526145449615057  p75=0.4182825715493033
tippy_tap_fraction median=0.31705137751303053
slip_ratio         median=0.20809201142404024
tracking_ratio     median=0.6851440019458537  (achieved/commanded vx, walking holds; n=93)
schedule_completion_rate=0.57
terminations={'fall': 43, 'schedule_complete': 57}

by hold:
     stand: cmd 0.00 -> achieved 0.121 m/s | tripod median=0.4156474126811819 (n=97)
     creep: cmd 0.25 -> achieved 0.215 m/s | tripod median=0.415637562859366 (n=93)
       low: cmd 0.35 -> achieved 0.159 m/s | tripod median=0.23357889040849694 (n=62)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
