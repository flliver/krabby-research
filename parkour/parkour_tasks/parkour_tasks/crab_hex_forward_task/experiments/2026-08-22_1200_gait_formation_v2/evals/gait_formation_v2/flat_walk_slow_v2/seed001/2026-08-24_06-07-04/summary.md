```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-24_00-10-08/model_9998.pt
episodes   : 100  unscored: 7
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.19510812694987975  p25=0.15591793778338034  p75=0.22914143642779142
tippy_tap_fraction median=0.3961038961038961
slip_ratio         median=0.6254015977593985
tracking_ratio     median=0.061931573815536045  (achieved/commanded vx, walking holds; n=85)
schedule_completion_rate=0.28
terminations={'schedule_complete': 28, 'fall': 72}

by hold:
     stand: cmd 0.00 -> achieved -0.002 m/s | tripod median=0.23562925743865953 (n=93)
     creep: cmd 0.25 -> achieved 0.008 m/s | tripod median=0.16594362645137056 (n=85)
       low: cmd 0.35 -> achieved 0.023 m/s | tripod median=0.10875514375416856 (n=51)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
