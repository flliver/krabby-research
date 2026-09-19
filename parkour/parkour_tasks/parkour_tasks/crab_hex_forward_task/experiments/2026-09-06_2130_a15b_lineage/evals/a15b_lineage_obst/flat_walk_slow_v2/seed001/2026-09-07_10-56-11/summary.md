```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-07_04-38-50/model_19996.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.489971185293664  p25=0.418747251256214  p75=0.5396498590295836
tippy_tap_fraction median=0.23337347785360632
slip_ratio         median=0.17743851100056363
tracking_ratio     median=0.5986623664197949  (achieved/commanded vx, walking holds; n=95)
schedule_completion_rate=0.66
terminations={'schedule_complete': 66, 'fall': 34}

by hold:
     stand: cmd 0.00 -> achieved 0.117 m/s | tripod median=0.44225659925468475 (n=100)
     creep: cmd 0.25 -> achieved 0.167 m/s | tripod median=0.512759087935202 (n=95)
       low: cmd 0.35 -> achieved 0.184 m/s | tripod median=0.5465091530481911 (n=77)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
