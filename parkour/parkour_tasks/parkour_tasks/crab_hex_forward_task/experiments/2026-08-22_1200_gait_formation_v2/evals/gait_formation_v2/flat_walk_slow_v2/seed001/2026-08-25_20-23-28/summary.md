```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-25_14-30-49/model_19996.pt
episodes   : 100  unscored: 6
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6039725948833158  p25=0.576606247117904  p75=0.6428876674400822
tippy_tap_fraction median=0.25
slip_ratio         median=0.29897380869038814
tracking_ratio     median=0.3866432266353936  (achieved/commanded vx, walking holds; n=89)
schedule_completion_rate=0.64
terminations={'schedule_complete': 64, 'fall': 36}

by hold:
     stand: cmd 0.00 -> achieved 0.028 m/s | tripod median=0.6176672345389698 (n=94)
     creep: cmd 0.25 -> achieved 0.111 m/s | tripod median=0.6535570787030084 (n=89)
       low: cmd 0.35 -> achieved 0.113 m/s | tripod median=0.5610047737671491 (n=66)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
