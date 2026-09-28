```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-31_03-42-16/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5946052676788929  p25=0.5505664649349993  p75=0.6335517700035109
tippy_tap_fraction median=0.3105590062111801
slip_ratio         median=0.3207306360212961
tracking_ratio     median=0.39697300320233286  (achieved/commanded vx, walking holds; n=92)
schedule_completion_rate=0.88
terminations={'schedule_complete': 88, 'fall': 12}

by hold:
     stand: cmd 0.00 -> achieved 0.048 m/s | tripod median=0.5520322469079828 (n=100)
     creep: cmd 0.25 -> achieved 0.110 m/s | tripod median=0.6345308938202661 (n=92)
       low: cmd 0.35 -> achieved 0.125 m/s | tripod median=0.630270497621976 (n=88)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
