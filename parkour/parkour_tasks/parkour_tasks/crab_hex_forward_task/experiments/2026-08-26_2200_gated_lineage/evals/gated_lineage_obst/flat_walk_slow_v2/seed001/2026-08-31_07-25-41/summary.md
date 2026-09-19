```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-31_01-12-23/model_19996.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.3990270230952434  p25=0.3292668494570342  p75=0.4511962576070121
tippy_tap_fraction median=0.2946778711484594
slip_ratio         median=0.26426352001703923
tracking_ratio     median=0.5425054060024623  (achieved/commanded vx, walking holds; n=97)
schedule_completion_rate=0.53
terminations={'schedule_complete': 53, 'fall': 47}

by hold:
     stand: cmd 0.00 -> achieved 0.015 m/s | tripod median=0.12543044001013565 (n=100)
     creep: cmd 0.25 -> achieved 0.154 m/s | tripod median=0.6191354597911372 (n=97)
       low: cmd 0.35 -> achieved 0.158 m/s | tripod median=0.5466104420248918 (n=64)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
