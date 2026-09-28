```
=== crab-hex gait eval ===
scenario   : step__base  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-31_10-46-19/model_19996.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5774458089629633  p25=0.5432638846664151  p75=0.6137111261274006
tippy_tap_fraction median=0.2341098841906814
slip_ratio         median=0.1827434541738756
tracking_ratio     median=0.46171464214993685  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.64
terminations={'schedule_complete': 64, 'fall': 36}

by hold:
     stand: cmd 0.00 -> achieved 0.073 m/s | tripod median=0.6679109513756817 (n=100)
     creep: cmd 0.25 -> achieved 0.129 m/s | tripod median=0.5022002487952093 (n=100)
       low: cmd 0.35 -> achieved 0.132 m/s | tripod median=0.5184845288975363 (n=71)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
