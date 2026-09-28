```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-05_16-51-20/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.523484998185332  p25=0.47676960982749134  p75=0.5723008294012261
tippy_tap_fraction median=0.23345498783454988
slip_ratio         median=0.21026524158160148
tracking_ratio     median=0.666416604993455  (achieved/commanded vx, walking holds; n=99)
schedule_completion_rate=0.68
terminations={'schedule_complete': 68, 'fall': 32}

by hold:
     stand: cmd 0.00 -> achieved 0.057 m/s | tripod median=0.44940673477386284 (n=100)
     creep: cmd 0.25 -> achieved 0.198 m/s | tripod median=0.5890088245423589 (n=99)
       low: cmd 0.35 -> achieved 0.186 m/s | tripod median=0.5536708005530663 (n=84)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
