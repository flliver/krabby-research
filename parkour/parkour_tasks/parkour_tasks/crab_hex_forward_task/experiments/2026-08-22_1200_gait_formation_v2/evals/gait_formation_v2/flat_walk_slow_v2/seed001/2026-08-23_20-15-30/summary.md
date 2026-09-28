```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-23_14-10-59/model_4999.pt
episodes   : 100  unscored: 10
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.24246846341164993  p25=0.11462293332002156  p75=0.32512925341639165
tippy_tap_fraction median=0.41975308641975306
slip_ratio         median=0.42650301768396687
tracking_ratio     median=0.5636078993928251  (achieved/commanded vx, walking holds; n=81)
schedule_completion_rate=0.26
terminations={'schedule_complete': 26, 'fall': 74}

by hold:
     stand: cmd 0.00 -> achieved -0.000 m/s | tripod median=0.0 (n=85)
     creep: cmd 0.25 -> achieved 0.158 m/s | tripod median=0.4392866404532745 (n=81)
       low: cmd 0.35 -> achieved 0.158 m/s | tripod median=0.49228405005394793 (n=44)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
