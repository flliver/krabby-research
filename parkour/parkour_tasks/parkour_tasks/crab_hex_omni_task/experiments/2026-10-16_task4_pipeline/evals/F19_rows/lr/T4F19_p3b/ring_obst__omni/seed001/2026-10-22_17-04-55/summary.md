```
=== crab-hex gait eval ===
scenario   : ring_obst__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7008324500389975  p25=0.6929203779606403  p75=0.7105685141907336
tippy_tap_fraction median=0.4299156108942629
slip_ratio         median=0.26398939715267683
tracking_ratio     median=0.9283966341333351  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}
facing_error_rms   median=0.5916796186052488  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.7893871347190453 (n=100)
       fwd: cmd 0.25 -> achieved 0.244 m/s | tripod median=0.578379017443649 (n=100)
     lat_l: cmd 0.22 -> achieved 0.225 m/s | tripod median=0.6435041236324289 (n=100)
      diag: cmd 0.25 -> achieved 0.249 m/s | tripod median=0.6590548289124034 (n=100)
      back: cmd 0.22 -> achieved 0.211 m/s | tripod median=0.7010721396339468 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.006 m/s | tripod median=0.7765164363979884 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.011 m/s | tripod median=0.7829331305562348 (n=100)
  move_turn: cmd 0.22 -> achieved 0.159 m/s | tripod median=0.6133344484659073 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.019 initial=0.005 final=0.020 max=0.046 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a
       fwd: err rms=0.056 initial=0.021 final=0.070 max=0.115 closed=n/a | settle=0.000 (n=97/100) | progress=-0.000 net=0.010 yaw_rate_ratio=0.392
     lat_l: err rms=0.049 initial=0.070 final=0.026 max=0.101 closed=0.798 | settle=0.000 (n=100/100) | progress=0.003 net=0.005 yaw_rate_ratio=0.457
      diag: err rms=0.046 initial=0.027 final=0.032 max=0.089 closed=n/a | settle=0.000 (n=99/100) | progress=-0.018 net=0.001 yaw_rate_ratio=0.413
      back: err rms=0.061 initial=0.031 final=0.085 max=0.120 closed=0.827 | settle=0.000 (n=95/100) | progress=-0.002 net=0.014 yaw_rate_ratio=0.617
  turn_l90: err rms=0.744 initial=1.485 final=0.022 max=1.490 closed=0.985 | settle=7.830 (n=100/100) | progress=0.170 net=0.123 yaw_rate_ratio=0.647
  turn_r90: err rms=0.777 initial=1.562 final=0.035 max=1.566 closed=0.977 | settle=8.100 (n=100/100) | progress=0.175 net=0.128 yaw_rate_ratio=0.649
  move_turn: err rms=1.216 initial=1.537 final=0.927 max=1.545 closed=0.395 | settle=n/a (n=0/100) | progress=0.074 net=0.075 yaw_rate_ratio=0.404

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
