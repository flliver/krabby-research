```
=== crab-hex gait eval ===
scenario   : slow__A10pB  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-31_10-46-19/model_19996.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6550238360247673  p25=0.6288602291618611  p75=0.6889506845347442
tippy_tap_fraction median=0.2354631507775524
slip_ratio         median=0.21196098189231521
tracking_ratio     median=0.39483467833876873  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}

by hold:
     stand: cmd 0.00 -> achieved 0.059 m/s | tripod median=0.6636547118022154 (n=100)
     creep: cmd 0.25 -> achieved 0.097 m/s | tripod median=0.6527005694110339 (n=100)
       low: cmd 0.35 -> achieved 0.140 m/s | tripod median=0.6736845859214722 (n=100)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
