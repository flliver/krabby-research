```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-24_19-06-17/model_14997.pt
episodes   : 100  unscored: 5
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.4734378327384113  p25=0.4263723943469512  p75=0.5168141259899683
tippy_tap_fraction median=0.24822695035460993
slip_ratio         median=0.26283616324048964
tracking_ratio     median=0.4506125057899178  (achieved/commanded vx, walking holds; n=95)
schedule_completion_rate=0.58
terminations={'fall': 42, 'schedule_complete': 58}

by hold:
     stand: cmd 0.00 -> achieved 0.006 m/s | tripod median=0.2840856652862616 (n=95)
     creep: cmd 0.25 -> achieved 0.135 m/s | tripod median=0.6359404107841651 (n=95)
       low: cmd 0.35 -> achieved 0.118 m/s | tripod median=0.5358785659968398 (n=71)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
