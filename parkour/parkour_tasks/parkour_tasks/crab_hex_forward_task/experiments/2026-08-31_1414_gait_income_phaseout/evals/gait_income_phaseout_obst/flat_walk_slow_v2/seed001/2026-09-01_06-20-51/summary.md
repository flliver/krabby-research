```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-31_23-49-32/model_9998.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5725655160396916  p25=0.5149715237643593  p75=0.6162011874739991
tippy_tap_fraction median=0.23876291039178352
slip_ratio         median=0.2123945473397699
tracking_ratio     median=0.5300498620700226  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.45
terminations={'schedule_complete': 45, 'fall': 55}

by hold:
     stand: cmd 0.00 -> achieved 0.045 m/s | tripod median=0.5631073051245428 (n=100)
     creep: cmd 0.25 -> achieved 0.142 m/s | tripod median=0.6372684892907556 (n=100)
       low: cmd 0.35 -> achieved 0.142 m/s | tripod median=0.4728910990452635 (n=58)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
