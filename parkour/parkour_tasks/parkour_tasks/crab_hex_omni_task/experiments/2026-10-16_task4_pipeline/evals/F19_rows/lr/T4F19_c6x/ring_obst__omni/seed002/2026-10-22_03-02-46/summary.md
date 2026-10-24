```
=== crab-hex gait eval ===
scenario   : ring_obst__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-10-21_20-59-46_seed19_X05_c6x_A15pB/model_14995.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.663242289928222  p25=0.6421288395223379  p75=0.6745207405167643
tippy_tap_fraction median=0.3903360570129631
slip_ratio         median=0.24828547943710927
tracking_ratio     median=0.893410092228762  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.96
terminations={'schedule_complete': 96, 'fall': 4}
facing_error_rms   median=0.6524660583175768  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.766558129544815 (n=100)
       fwd: cmd 0.25 -> achieved 0.243 m/s | tripod median=0.5556712960079666 (n=100)
     lat_l: cmd 0.22 -> achieved 0.224 m/s | tripod median=0.7845519370097982 (n=100)
      diag: cmd 0.25 -> achieved 0.242 m/s | tripod median=0.6573788715710104 (n=100)
      back: cmd 0.22 -> achieved 0.211 m/s | tripod median=0.5307225018197614 (n=100)
  turn_l90: cmd 0.00 -> achieved 0.012 m/s | tripod median=0.7249052093166456 (n=100)
  turn_r90: cmd 0.00 -> achieved -0.000 m/s | tripod median=0.7519848192430587 (n=100)
  move_turn: cmd 0.22 -> achieved 0.126 m/s | tripod median=0.4639306876382713 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.019 initial=0.010 final=0.017 max=0.042 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.050 initial=0.016 final=0.041 max=0.101 closed=n/a | settle=0.000 (n=100/100) | progress=0.003 net=0.005 yaw_rate_ratio=0.319
     lat_l: err rms=0.036 initial=0.043 final=0.017 max=0.078 closed=n/a | settle=0.000 (n=100/100) | progress=-0.007 net=0.005 yaw_rate_ratio=0.485
      diag: err rms=0.040 initial=0.017 final=0.041 max=0.085 closed=n/a | settle=0.000 (n=100/100) | progress=0.009 net=0.001 yaw_rate_ratio=1.023
      back: err rms=0.056 initial=0.042 final=0.079 max=0.115 closed=n/a | settle=0.000 (n=98/100) | progress=-0.008 net=0.016 yaw_rate_ratio=0.574
  turn_l90: err rms=1.250 initial=1.651 final=0.864 max=1.661 closed=0.478 | settle=n/a (n=0/100) | progress=0.066 net=0.066 yaw_rate_ratio=0.417
  turn_r90: err rms=0.297 initial=0.707 final=0.023 max=0.721 closed=0.965 | settle=5.120 (n=99/100) | progress=0.110 net=0.058 yaw_rate_ratio=0.443
  move_turn: err rms=1.206 initial=1.554 final=0.964 max=1.558 closed=0.385 | settle=n/a (n=0/100) | progress=0.076 net=0.077 yaw_rate_ratio=0.419

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
