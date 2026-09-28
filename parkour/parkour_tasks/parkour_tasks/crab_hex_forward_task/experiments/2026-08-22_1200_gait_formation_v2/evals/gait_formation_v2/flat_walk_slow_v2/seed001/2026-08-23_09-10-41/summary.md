```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-23_03-17-24/model_4999.pt
episodes   : 100  unscored: 3
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.49561380662564664  p25=0.459104556141771  p75=0.5425275604220017
tippy_tap_fraction median=0.27164061942481976
slip_ratio         median=0.29456799367885833
tracking_ratio     median=0.45336109834859833  (achieved/commanded vx, walking holds; n=96)
schedule_completion_rate=0.94
terminations={'schedule_complete': 94, 'fall': 6}

by hold:
     stand: cmd 0.00 -> achieved 0.016 m/s | tripod median=0.33045118792670564 (n=97)
     creep: cmd 0.25 -> achieved 0.136 m/s | tripod median=0.6209294768007596 (n=96)
       low: cmd 0.35 -> achieved 0.129 m/s | tripod median=0.5561904806475707 (n=95)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
