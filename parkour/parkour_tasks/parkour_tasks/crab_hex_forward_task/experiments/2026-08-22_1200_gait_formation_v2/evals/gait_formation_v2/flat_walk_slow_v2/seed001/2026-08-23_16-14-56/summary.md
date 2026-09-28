```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-23_10-17-59/model_4999.pt
episodes   : 100  unscored: 6
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.25563413755617237  p25=0.21710550813293727  p75=0.27798752713971936
tippy_tap_fraction median=0.34415584415584416
slip_ratio         median=0.3402255887266143
tracking_ratio     median=0.5235783624317031  (achieved/commanded vx, walking holds; n=90)
schedule_completion_rate=0.63
terminations={'schedule_complete': 63, 'fall': 37}

by hold:
     stand: cmd 0.00 -> achieved 0.008 m/s | tripod median=0.027366443223622326 (n=94)
     creep: cmd 0.25 -> achieved 0.142 m/s | tripod median=0.39758409271717926 (n=90)
       low: cmd 0.35 -> achieved 0.166 m/s | tripod median=0.36299376208326173 (n=75)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
