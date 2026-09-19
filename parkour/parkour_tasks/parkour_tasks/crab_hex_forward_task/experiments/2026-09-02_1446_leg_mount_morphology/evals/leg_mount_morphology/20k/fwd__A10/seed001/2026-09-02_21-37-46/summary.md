```
=== crab-hex gait eval ===
scenario   : fwd__A10  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-31_10-46-19/model_19996.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5197112195204372  p25=0.5000171743310963  p75=0.5402536845910982
tippy_tap_fraction median=0.28484848484848485
slip_ratio         median=0.2628329930820147
tracking_ratio     median=0.3087334067548225  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}

by hold:
       low: cmd 0.30 -> achieved 0.128 m/s | tripod median=0.6073066104921612 (n=100)
       mid: cmd 0.47 -> achieved 0.132 m/s | tripod median=0.5192835192122768 (n=100)
      high: cmd 0.65 -> achieved 0.146 m/s | tripod median=0.42194376420029955 (n=100)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
