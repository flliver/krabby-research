```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-24_13-13-08/model_19996.pt
episodes   : 100  unscored: 4
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5140524939262664  p25=0.45470619556383235  p75=0.5465022023005587
tippy_tap_fraction median=0.2472298343857977
slip_ratio         median=0.2580271386133005
tracking_ratio     median=0.5664641940400406  (achieved/commanded vx, walking holds; n=95)
schedule_completion_rate=0.27
terminations={'fall': 73, 'schedule_complete': 27}

by hold:
     stand: cmd 0.00 -> achieved 0.022 m/s | tripod median=0.5419633983535647 (n=96)
     creep: cmd 0.25 -> achieved 0.146 m/s | tripod median=0.5547895274859805 (n=95)
       low: cmd 0.35 -> achieved 0.181 m/s | tripod median=0.4151520536680752 (n=64)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
