```
=== crab-hex gait eval ===
scenario   : step__side  (Isaac-Crab-Hex-Side-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_side_teacher/2026-09-12_03-04-37/model_9998.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.3307568474179612  p25=0.24569572659173805  p75=0.4159543344314337
tippy_tap_fraction median=0.389027698458024
slip_ratio         median=0.49082606780849625
tracking_ratio     median=0.5696726919879769  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.88
terminations={'schedule_complete': 88, 'fall': 12}

by hold:
     stand: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.0 (n=92)
     creep: cmd 0.25 -> achieved 0.199 m/s | tripod median=0.5793772360500178 (n=100)
       low: cmd 0.35 -> achieved 0.132 m/s | tripod median=0.369737489136429 (n=95)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
