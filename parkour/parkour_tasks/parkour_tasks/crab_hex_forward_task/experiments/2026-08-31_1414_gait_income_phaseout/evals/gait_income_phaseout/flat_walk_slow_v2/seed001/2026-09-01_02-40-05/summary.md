```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-31_21-08-05/model_9997.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5949510779491372  p25=0.5614103523531391  p75=0.6274036308744453
tippy_tap_fraction median=0.24117158288325719
slip_ratio         median=0.21793572171691444
tracking_ratio     median=0.4213219010694222  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.97
terminations={'schedule_complete': 97, 'fall': 3}

by hold:
     stand: cmd 0.00 -> achieved 0.042 m/s | tripod median=0.5596739416857068 (n=100)
     creep: cmd 0.25 -> achieved 0.117 m/s | tripod median=0.6678349934567847 (n=100)
       low: cmd 0.35 -> achieved 0.130 m/s | tripod median=0.56277856387179 (n=99)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
