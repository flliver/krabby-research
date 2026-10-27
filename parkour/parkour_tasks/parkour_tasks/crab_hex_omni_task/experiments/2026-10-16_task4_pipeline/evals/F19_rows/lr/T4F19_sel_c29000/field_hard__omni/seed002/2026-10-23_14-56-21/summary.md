```
=== crab-hex gait eval ===
scenario   : field_hard__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_29000.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6301559173832072  p25=0.5982724609983874  p75=0.6631818828391574
tippy_tap_fraction median=0.3939978049115105
slip_ratio         median=0.2769527819030212
tracking_ratio     median=0.8712142363189969  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.87
terminations={'schedule_complete': 87, 'fall': 13}
facing_error_rms   median=0.5988428061883927  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7740583296701042 (n=100)
       fwd: cmd 0.25 -> achieved 0.239 m/s | tripod median=0.57364845291737 (n=100)
     lat_l: cmd 0.22 -> achieved 0.209 m/s | tripod median=0.5553319299720586 (n=100)
      diag: cmd 0.25 -> achieved 0.232 m/s | tripod median=0.5520245603353511 (n=97)
      back: cmd 0.22 -> achieved 0.208 m/s | tripod median=0.6161486860712959 (n=94)
  turn_l90: cmd 0.00 -> achieved -0.009 m/s | tripod median=0.7362466084829562 (n=94)
  turn_r90: cmd 0.00 -> achieved 0.007 m/s | tripod median=0.7441505518180629 (n=94)
  move_turn: cmd 0.22 -> achieved 0.147 m/s | tripod median=0.5194604369699891 (n=93)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.021 initial=0.006 final=0.024 max=0.045 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a
       fwd: err rms=0.063 initial=0.023 final=0.073 max=0.138 closed=n/a | settle=0.000 (n=88/100) | progress=-0.002 net=0.009 yaw_rate_ratio=0.511
     lat_l: err rms=0.058 initial=0.075 final=0.042 max=0.113 closed=0.701 | settle=0.000 (n=100/100) | progress=0.008 net=0.005 yaw_rate_ratio=0.546
      diag: err rms=0.054 initial=0.042 final=0.051 max=0.099 closed=n/a | settle=0.000 (n=89/99) | progress=-0.007 net=0.003 yaw_rate_ratio=0.521
      back: err rms=0.070 initial=0.053 final=0.078 max=0.132 closed=0.760 | settle=0.000 (n=93/94) | progress=-0.000 net=0.013 yaw_rate_ratio=0.543
  turn_l90: err rms=0.781 initial=1.497 final=0.030 max=1.499 closed=0.980 | settle=8.320 (n=92/94) | progress=0.164 net=0.122 yaw_rate_ratio=0.630
  turn_r90: err rms=0.780 initial=1.547 final=0.040 max=1.549 closed=0.974 | settle=8.210 (n=94/94) | progress=0.170 net=0.126 yaw_rate_ratio=0.640
  move_turn: err rms=1.235 initial=1.534 final=0.965 max=1.540 closed=0.381 | settle=n/a (n=0/94) | progress=0.073 net=0.074 yaw_rate_ratio=0.421

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
