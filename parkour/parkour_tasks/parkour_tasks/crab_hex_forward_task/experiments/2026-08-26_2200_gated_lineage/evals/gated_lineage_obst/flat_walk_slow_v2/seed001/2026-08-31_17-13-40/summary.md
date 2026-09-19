```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-31_10-46-19/model_19996.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5889223062215885  p25=0.5473432618860488  p75=0.6248259397824549
tippy_tap_fraction median=0.2238530556966311
slip_ratio         median=0.17675401247048655
tracking_ratio     median=0.49380160728540484  (achieved/commanded vx, walking holds; n=99)
schedule_completion_rate=0.48
terminations={'schedule_complete': 48, 'fall': 52}

by hold:
     stand: cmd 0.00 -> achieved 0.073 m/s | tripod median=0.6580648111635394 (n=100)
     creep: cmd 0.25 -> achieved 0.137 m/s | tripod median=0.5231537243559002 (n=99)
       low: cmd 0.35 -> achieved 0.145 m/s | tripod median=0.5142202555355413 (n=58)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
