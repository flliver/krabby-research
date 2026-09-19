```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-22_15-04-05/model_4999.pt
episodes   : 100  unscored: 3
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.2568596019087625  p25=0.16743678016212846  p75=0.30614634970469523
tippy_tap_fraction median=0.3984196806777452
slip_ratio         median=0.43513425306144604
tracking_ratio     median=0.5019220542338078  (achieved/commanded vx, walking holds; n=87)
schedule_completion_rate=0.2
terminations={'fall': 80, 'schedule_complete': 20}

by hold:
     stand: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.06869294463982778 (n=97)
     creep: cmd 0.25 -> achieved 0.135 m/s | tripod median=0.4289611258725685 (n=86)
       low: cmd 0.35 -> achieved 0.150 m/s | tripod median=0.34619817828077826 (n=50)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
