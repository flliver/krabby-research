```
=== crab-hex gait eval ===
scenario   : step__A15  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-03_04-53-36/model_4999.pt
episodes   : 100  unscored: 4
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.2325213680585581  p25=0.08543388658737927  p75=0.31510683749452545
tippy_tap_fraction median=0.4203373377017352
slip_ratio         median=0.35608167967666865
tracking_ratio     median=0.5690048960677567  (achieved/commanded vx, walking holds; n=75)
schedule_completion_rate=0.32
terminations={'fall': 68, 'schedule_complete': 32}

by hold:
     stand: cmd 0.00 -> achieved 0.008 m/s | tripod median=0.016760183269727676 (n=96)
     creep: cmd 0.25 -> achieved 0.148 m/s | tripod median=0.4478386233521866 (n=75)
       low: cmd 0.35 -> achieved 0.179 m/s | tripod median=0.4199004075018894 (n=52)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
