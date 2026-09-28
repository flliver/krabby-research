```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-25_12-33-44/model_19996.pt
episodes   : 100  unscored: 5
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5690450308304144  p25=0.5278249689306551  p75=0.6011692135573999
tippy_tap_fraction median=0.2236024844720497
slip_ratio         median=0.27889369698267763
tracking_ratio     median=0.4027151296717828  (achieved/commanded vx, walking holds; n=95)
schedule_completion_rate=0.81
terminations={'schedule_complete': 81, 'fall': 19}

by hold:
     stand: cmd 0.00 -> achieved 0.026 m/s | tripod median=0.5719737270348013 (n=95)
     creep: cmd 0.25 -> achieved 0.109 m/s | tripod median=0.6278073428271489 (n=95)
       low: cmd 0.35 -> achieved 0.125 m/s | tripod median=0.5163606871359816 (n=87)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
