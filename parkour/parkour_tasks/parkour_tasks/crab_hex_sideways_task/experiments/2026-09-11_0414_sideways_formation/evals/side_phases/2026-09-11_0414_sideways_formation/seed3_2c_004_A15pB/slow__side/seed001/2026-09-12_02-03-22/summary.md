```
=== crab-hex gait eval ===
scenario   : slow__side  (Isaac-Crab-Hex-Side-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_side_teacher/2026-09-11_19-43-19/model_19996.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.3635271131866803  p25=0.31800717293811165  p75=0.4108018752021733
tippy_tap_fraction median=0.41397412450044024
slip_ratio         median=0.40529220113418846
tracking_ratio     median=0.6592240902151469  (achieved/commanded vx, walking holds; n=95)
schedule_completion_rate=0.84
terminations={'schedule_complete': 84, 'fall': 16}

by hold:
     stand: cmd 0.00 -> achieved 0.006 m/s | tripod median=0.0 (n=92)
     creep: cmd 0.25 -> achieved 0.197 m/s | tripod median=0.5674172430636106 (n=95)
       low: cmd 0.35 -> achieved 0.183 m/s | tripod median=0.48403451853262075 (n=87)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
