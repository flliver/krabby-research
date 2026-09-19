```
=== crab-hex gait eval ===
scenario   : slow__A15pB  (Isaac-Crab-Hex-Student-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_student/2026-09-08_05-54-01/model_24995.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5120929968270858  p25=0.4657095767319176  p75=0.5514795270962234
tippy_tap_fraction median=0.26490066225165565
slip_ratio         median=0.2074747047344337
tracking_ratio     median=0.6497648854702043  (achieved/commanded vx, walking holds; n=99)
schedule_completion_rate=0.79
terminations={'schedule_complete': 79, 'fall': 21}

by hold:
     stand: cmd 0.00 -> achieved 0.130 m/s | tripod median=0.5128191795986625 (n=100)
     creep: cmd 0.25 -> achieved 0.182 m/s | tripod median=0.5398415149445926 (n=99)
       low: cmd 0.35 -> achieved 0.194 m/s | tripod median=0.49941881810824007 (n=88)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
