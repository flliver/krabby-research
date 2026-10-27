```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_side  (Isaac-Crab-Hex-Side-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_side_teacher/2026-09-12_05-24-07/model_14997.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.39394033988559285  p25=0.349290574626949  p75=0.43862246226170853
tippy_tap_fraction median=0.3506756756756757
slip_ratio         median=0.4557755755071464
tracking_ratio     median=0.6661458295129102  (achieved/commanded vx, walking holds; n=98)
schedule_completion_rate=0.84
terminations={'schedule_complete': 84, 'fall': 16}

by hold:
     stand: cmd 0.00 -> achieved 0.005 m/s | tripod median=0.0 (n=93)
     creep: cmd 0.25 -> achieved 0.204 m/s | tripod median=0.6220672051768366 (n=98)
       low: cmd 0.35 -> achieved 0.196 m/s | tripod median=0.5684077857381291 (n=87)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
