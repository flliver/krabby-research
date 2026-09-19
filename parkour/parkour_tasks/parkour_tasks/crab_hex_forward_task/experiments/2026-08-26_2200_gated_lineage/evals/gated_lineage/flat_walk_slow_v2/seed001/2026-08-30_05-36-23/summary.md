```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-29_23-24-23/model_9998.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5016225356250519  p25=0.4840299445277133  p75=0.5394868139720114
tippy_tap_fraction median=0.24853781167698463
slip_ratio         median=0.2540721783379936
tracking_ratio     median=0.49386664159772287  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}

by hold:
     stand: cmd 0.00 -> achieved 0.028 m/s | tripod median=0.29164130426486984 (n=100)
     creep: cmd 0.25 -> achieved 0.133 m/s | tripod median=0.5799528025410436 (n=100)
       low: cmd 0.35 -> achieved 0.160 m/s | tripod median=0.6582458052465503 (n=100)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
