```
=== crab-hex gait eval ===
scenario   : fwd__base  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-31_03-42-16/model_4999.pt
episodes   : 100  unscored: 12
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5577640992214865  p25=0.5242015314127135  p75=0.587840736122518
tippy_tap_fraction median=0.40544462502902257
slip_ratio         median=0.3059280010966973
tracking_ratio     median=0.3910912920781373  (achieved/commanded vx, walking holds; n=89)
schedule_completion_rate=0.0
terminations={'fall': 100}

by hold:
       low: cmd 0.30 -> achieved 0.151 m/s | tripod median=0.683607933498696 (n=88)
       mid: cmd 0.47 -> achieved 0.133 m/s | tripod median=0.43608887153035364 (n=71)
      high: cmd 0.65 -> achieved 0.184 m/s | tripod median=0.31753248810651513 (n=36)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
