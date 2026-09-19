```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-23_01-11-03/model_4999.pt
episodes   : 100  unscored: 3
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.0  p25=0.0  p75=0.01183541335136795
tippy_tap_fraction median=0.42846826606755994
slip_ratio         median=1.1434091375346582
tracking_ratio     median=0.12028142276729915  (achieved/commanded vx, walking holds; n=96)
schedule_completion_rate=0.29
terminations={'fall': 71, 'schedule_complete': 29}

by hold:
     stand: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.0 (n=89)
     creep: cmd 0.25 -> achieved 0.042 m/s | tripod median=0.0 (n=95)
       low: cmd 0.35 -> achieved 0.013 m/s | tripod median=0.0 (n=56)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
