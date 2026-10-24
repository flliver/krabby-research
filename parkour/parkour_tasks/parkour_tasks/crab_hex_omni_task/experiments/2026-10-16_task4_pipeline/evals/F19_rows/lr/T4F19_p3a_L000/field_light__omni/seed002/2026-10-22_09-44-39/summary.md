```
=== crab-hex gait eval ===
scenario   : field_light__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_03-27-53_seed19_X07_p3a_A15pB/model_22993.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7709482975662885  p25=0.7610874320377432  p75=0.7788726885951864
tippy_tap_fraction median=0.41142171154175955
slip_ratio         median=0.2405841593132872
tracking_ratio     median=0.9312883271535048  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.99
terminations={'schedule_complete': 99, 'fall': 1}
facing_error_rms   median=0.5975290106458044  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.8334746806745863 (n=100)
       fwd: cmd 0.25 -> achieved 0.251 m/s | tripod median=0.7713763182523137 (n=100)
     lat_l: cmd 0.22 -> achieved 0.204 m/s | tripod median=0.7329571557858361 (n=100)
      diag: cmd 0.25 -> achieved 0.236 m/s | tripod median=0.6938235567808757 (n=100)
      back: cmd 0.22 -> achieved 0.226 m/s | tripod median=0.7425625086331957 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.009 m/s | tripod median=0.8187576013974799 (n=99)
  turn_r90: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.8084200517014029 (n=99)
  move_turn: cmd 0.22 -> achieved 0.165 m/s | tripod median=0.7289081296996474 (n=99)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.020 initial=0.009 final=0.020 max=0.047 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.000 yaw_rate_ratio=n/a
       fwd: err rms=0.057 initial=0.021 final=0.106 max=0.117 closed=n/a | settle=0.000 (n=92/100) | progress=-0.010 net=-0.001 yaw_rate_ratio=0.282
     lat_l: err rms=0.063 initial=0.107 final=0.019 max=0.130 closed=0.923 | settle=0.000 (n=100/100) | progress=0.006 net=0.010 yaw_rate_ratio=0.418
      diag: err rms=0.032 initial=0.018 final=0.023 max=0.073 closed=n/a | settle=0.000 (n=100/100) | progress=-0.004 net=0.001 yaw_rate_ratio=0.349
      back: err rms=0.062 initial=0.024 final=0.098 max=0.126 closed=n/a | settle=0.000 (n=95/100) | progress=-0.004 net=0.013 yaw_rate_ratio=0.394
  turn_l90: err rms=0.972 initial=1.668 final=0.129 max=1.690 closed=0.923 | settle=11.540 (n=73/99) | progress=0.133 net=0.128 yaw_rate_ratio=0.586
  turn_r90: err rms=0.710 initial=1.446 final=0.120 max=1.458 closed=0.916 | settle=10.860 (n=88/99) | progress=0.128 net=0.110 yaw_rate_ratio=0.587
  move_turn: err rms=1.048 initial=1.452 final=0.617 max=1.468 closed=0.568 | settle=n/a (n=0/99) | progress=0.102 net=0.102 yaw_rate_ratio=0.422

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
