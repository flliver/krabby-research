```
=== crab-hex gait eval ===
scenario   : slow__A15pB  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-07_09-16-39/model_29994.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.37008774266748556  p25=0.31918789051203605  p75=0.4182377767018472
tippy_tap_fraction median=0.24295125164690384
slip_ratio         median=0.1960913561972169
tracking_ratio     median=0.6951062849124119  (achieved/commanded vx, walking holds; n=97)
schedule_completion_rate=0.59
terminations={'fall': 41, 'schedule_complete': 59}

by hold:
     stand: cmd 0.00 -> achieved 0.121 m/s | tripod median=0.2926518958298714 (n=100)
     creep: cmd 0.25 -> achieved 0.201 m/s | tripod median=0.42987801660239255 (n=97)
       low: cmd 0.35 -> achieved 0.188 m/s | tripod median=0.40029426982872285 (n=71)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
