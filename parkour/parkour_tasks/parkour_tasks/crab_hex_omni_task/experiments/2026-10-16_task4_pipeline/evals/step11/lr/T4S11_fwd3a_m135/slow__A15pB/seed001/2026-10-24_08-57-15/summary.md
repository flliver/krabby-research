```
=== crab-hex gait eval ===
scenario   : slow__A15pB  (Isaac-Crab-Hex-Student-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_student/2026-09-08_05-54-01/model_24995.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.42335894923848605  p25=0.39473597092371904  p75=0.44326332275078467
tippy_tap_fraction median=0.3102686461647225
slip_ratio         median=0.29333516508544544
tracking_ratio     median=0.6141682107686046  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.61
terminations={'schedule_complete': 61, 'fall': 39}

by hold:
     stand: cmd 0.00 -> achieved 0.156 m/s | tripod median=0.5440469316659484 (n=100)
     creep: cmd 0.25 -> achieved 0.192 m/s | tripod median=0.3980444698911695 (n=100)
       low: cmd 0.35 -> achieved 0.155 m/s | tripod median=0.3011732223127043 (n=89)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
