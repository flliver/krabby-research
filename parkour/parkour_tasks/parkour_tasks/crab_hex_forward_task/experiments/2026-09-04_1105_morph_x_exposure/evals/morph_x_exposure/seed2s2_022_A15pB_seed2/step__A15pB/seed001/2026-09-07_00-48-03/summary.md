```
=== crab-hex gait eval ===
scenario   : step__A15pB  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-06_18-32-32/model_9998.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5596963040017826  p25=0.497854235588207  p75=0.6376000896921963
tippy_tap_fraction median=0.21192982456140352
slip_ratio         median=0.17474879264643917
tracking_ratio     median=0.6509179546839405  (achieved/commanded vx, walking holds; n=99)
schedule_completion_rate=0.92
terminations={'schedule_complete': 92, 'fall': 8}

by hold:
     stand: cmd 0.00 -> achieved 0.070 m/s | tripod median=0.5605544623341181 (n=100)
     creep: cmd 0.25 -> achieved 0.186 m/s | tripod median=0.5668008696815873 (n=99)
       low: cmd 0.35 -> achieved 0.199 m/s | tripod median=0.6242561771025841 (n=92)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
