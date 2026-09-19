```
=== crab-hex gait eval ===
scenario   : step__A15pB  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-06_23-59-54/model_9998.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.4186360780459436  p25=0.356695965489679  p75=0.4873765128856888
tippy_tap_fraction median=0.2599989362833741
slip_ratio         median=0.2339665397736726
tracking_ratio     median=0.6010067599261115  (achieved/commanded vx, walking holds; n=99)
schedule_completion_rate=0.74
terminations={'fall': 26, 'schedule_complete': 74}

by hold:
     stand: cmd 0.00 -> achieved 0.091 m/s | tripod median=0.36378309352241905 (n=100)
     creep: cmd 0.25 -> achieved 0.175 m/s | tripod median=0.46509426553840627 (n=98)
       low: cmd 0.35 -> achieved 0.174 m/s | tripod median=0.46327032978828653 (n=81)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
