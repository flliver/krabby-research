```
=== crab-hex gait eval ===
scenario   : ring_obst__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_03-27-53_seed19_X07_p3a_A15pB/model_22993.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7625648245823843  p25=0.7521048476685089  p75=0.7752963893666922
tippy_tap_fraction median=0.41569649384053065
slip_ratio         median=0.246242927677022
tracking_ratio     median=0.9291834900745339  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}
facing_error_rms   median=0.5963528445306704  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.000 m/s | tripod median=0.8339707647765134 (n=100)
       fwd: cmd 0.25 -> achieved 0.252 m/s | tripod median=0.7746498600634026 (n=100)
     lat_l: cmd 0.22 -> achieved 0.203 m/s | tripod median=0.73378018521708 (n=100)
      diag: cmd 0.25 -> achieved 0.237 m/s | tripod median=0.6876795649218953 (n=100)
      back: cmd 0.22 -> achieved 0.226 m/s | tripod median=0.7363918301681864 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.008 m/s | tripod median=0.8094866669617395 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7995282441368398 (n=100)
  move_turn: cmd 0.22 -> achieved 0.165 m/s | tripod median=0.7169576361236694 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.021 initial=0.011 final=0.023 max=0.051 closed=n/a | settle=0.000 (n=100/100) | progress=-0.007 net=0.001 yaw_rate_ratio=0.819
       fwd: err rms=0.063 initial=0.023 final=0.109 max=0.127 closed=n/a | settle=0.000 (n=94/100) | progress=-0.010 net=-0.002 yaw_rate_ratio=0.296
     lat_l: err rms=0.063 initial=0.110 final=0.023 max=0.133 closed=0.836 | settle=0.000 (n=100/100) | progress=0.006 net=0.011 yaw_rate_ratio=0.378
      diag: err rms=0.031 initial=0.023 final=0.018 max=0.072 closed=n/a | settle=0.000 (n=100/100) | progress=0.006 net=0.003 yaw_rate_ratio=0.323
      back: err rms=0.056 initial=0.018 final=0.094 max=0.122 closed=n/a | settle=0.000 (n=95/100) | progress=-0.007 net=0.010 yaw_rate_ratio=0.388
  turn_l90: err rms=0.959 initial=1.665 final=0.127 max=1.692 closed=0.924 | settle=11.470 (n=74/100) | progress=0.136 net=0.129 yaw_rate_ratio=0.589
  turn_r90: err rms=0.710 initial=1.444 final=0.118 max=1.453 closed=0.917 | settle=10.900 (n=90/100) | progress=0.128 net=0.111 yaw_rate_ratio=0.580
  move_turn: err rms=1.050 initial=1.451 final=0.628 max=1.462 closed=0.565 | settle=n/a (n=0/100) | progress=0.102 net=0.102 yaw_rate_ratio=0.421

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
