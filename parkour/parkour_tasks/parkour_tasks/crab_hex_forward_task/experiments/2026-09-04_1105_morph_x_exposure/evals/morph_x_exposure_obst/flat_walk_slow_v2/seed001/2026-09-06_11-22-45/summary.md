```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-06_05-05-22/model_9998.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5011450356919256  p25=0.4325692767359208  p75=0.5525079023366046
tippy_tap_fraction median=0.21603185381813694
slip_ratio         median=0.17606069959628157
tracking_ratio     median=0.5938280621339994  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.67
terminations={'schedule_complete': 67, 'fall': 33}

by hold:
     stand: cmd 0.00 -> achieved 0.050 m/s | tripod median=0.4353690823464086 (n=100)
     creep: cmd 0.25 -> achieved 0.162 m/s | tripod median=0.5587253833245407 (n=100)
       low: cmd 0.35 -> achieved 0.183 m/s | tripod median=0.529103571255709 (n=80)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
