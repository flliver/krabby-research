```
=== crab-hex gait eval ===
scenario   : ring_obst__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_03-27-53_seed19_X07_p3a_A15pB/model_22993.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7636507554270799  p25=0.754188032081479  p75=0.7763503594438537
tippy_tap_fraction median=0.4124629080118694
slip_ratio         median=0.24473633220401114
tracking_ratio     median=0.9312460476537676  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}
facing_error_rms   median=0.5960900281944046  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.8321666237021319 (n=100)
       fwd: cmd 0.25 -> achieved 0.251 m/s | tripod median=0.7691063003140484 (n=100)
     lat_l: cmd 0.22 -> achieved 0.204 m/s | tripod median=0.736712109502387 (n=100)
      diag: cmd 0.25 -> achieved 0.236 m/s | tripod median=0.6815305740576534 (n=100)
      back: cmd 0.22 -> achieved 0.227 m/s | tripod median=0.7448692450112718 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.009 m/s | tripod median=0.8148200479782388 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.795544357411868 (n=100)
  move_turn: cmd 0.22 -> achieved 0.166 m/s | tripod median=0.7220557832884056 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.021 initial=0.012 final=0.020 max=0.048 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.060 initial=0.020 final=0.101 max=0.119 closed=n/a | settle=0.000 (n=96/100) | progress=-0.003 net=-0.004 yaw_rate_ratio=0.275
     lat_l: err rms=0.067 initial=0.102 final=0.023 max=0.125 closed=0.882 | settle=0.000 (n=100/100) | progress=0.007 net=0.010 yaw_rate_ratio=0.401
      diag: err rms=0.034 initial=0.021 final=0.021 max=0.074 closed=n/a | settle=0.000 (n=100/100) | progress=-0.000 net=0.002 yaw_rate_ratio=0.404
      back: err rms=0.061 initial=0.020 final=0.097 max=0.126 closed=n/a | settle=0.000 (n=98/100) | progress=-0.003 net=0.012 yaw_rate_ratio=0.294
  turn_l90: err rms=0.963 initial=1.669 final=0.118 max=1.689 closed=0.929 | settle=11.520 (n=77/100) | progress=0.136 net=0.129 yaw_rate_ratio=0.585
  turn_r90: err rms=0.714 initial=1.452 final=0.117 max=1.458 closed=0.920 | settle=10.960 (n=90/100) | progress=0.128 net=0.111 yaw_rate_ratio=0.583
  move_turn: err rms=1.046 initial=1.454 final=0.616 max=1.465 closed=0.575 | settle=n/a (n=0/100) | progress=0.104 net=0.104 yaw_rate_ratio=0.422

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
