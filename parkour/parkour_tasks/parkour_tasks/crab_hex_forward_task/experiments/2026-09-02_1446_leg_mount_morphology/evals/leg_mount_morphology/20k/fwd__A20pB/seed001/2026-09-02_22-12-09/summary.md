```
=== crab-hex gait eval ===
scenario   : fwd__A20pB  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-31_10-46-19/model_19996.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5988980119954358  p25=0.5773494590223871  p75=0.6212967205134459
tippy_tap_fraction median=0.2742298553503689
slip_ratio         median=0.23364902458159695
tracking_ratio     median=0.32014014928858736  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}

by hold:
       low: cmd 0.30 -> achieved 0.120 m/s | tripod median=0.6576040862584152 (n=100)
       mid: cmd 0.47 -> achieved 0.144 m/s | tripod median=0.5913145449284021 (n=100)
      high: cmd 0.65 -> achieved 0.155 m/s | tripod median=0.5487736022287399 (n=100)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
