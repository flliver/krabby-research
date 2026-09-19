```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-23_22-06-56/model_4999.pt
episodes   : 100  unscored: 7
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.19911787613691573  p25=0.1611967746710218  p75=0.2368247706447582
tippy_tap_fraction median=0.3746666666666667
slip_ratio         median=0.5990767535713393
tracking_ratio     median=0.07483509065900257  (achieved/commanded vx, walking holds; n=86)
schedule_completion_rate=0.25
terminations={'fall': 75, 'schedule_complete': 25}

by hold:
     stand: cmd 0.00 -> achieved -0.000 m/s | tripod median=0.2562210049929085 (n=93)
     creep: cmd 0.25 -> achieved 0.010 m/s | tripod median=0.1917523784301831 (n=86)
       low: cmd 0.35 -> achieved 0.039 m/s | tripod median=0.11905116925519471 (n=67)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
