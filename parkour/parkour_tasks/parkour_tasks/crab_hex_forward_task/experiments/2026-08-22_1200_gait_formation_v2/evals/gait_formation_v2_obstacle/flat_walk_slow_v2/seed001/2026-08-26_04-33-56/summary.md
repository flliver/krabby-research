```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-25_19-09-15/model_24900.pt
episodes   : 100  unscored: 6
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5631282240513057  p25=0.4992855289973803  p75=0.623182508443298
tippy_tap_fraction median=0.21804511278195488
slip_ratio         median=0.22764013992846577
tracking_ratio     median=0.4162575518831315  (achieved/commanded vx, walking holds; n=94)
schedule_completion_rate=0.57
terminations={'fall': 43, 'schedule_complete': 57}

by hold:
     stand: cmd 0.00 -> achieved 0.026 m/s | tripod median=0.5477637401981522 (n=94)
     creep: cmd 0.25 -> achieved 0.118 m/s | tripod median=0.6219134851219966 (n=94)
       low: cmd 0.35 -> achieved 0.124 m/s | tripod median=0.5256682568446984 (n=72)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
