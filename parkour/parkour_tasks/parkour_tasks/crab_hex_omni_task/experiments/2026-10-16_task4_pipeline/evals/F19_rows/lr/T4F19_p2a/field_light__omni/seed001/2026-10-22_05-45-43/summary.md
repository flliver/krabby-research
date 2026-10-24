```
=== crab-hex gait eval ===
scenario   : field_light__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-21_23-06-12_seed19_X06_p2a_A15pB/model_19994.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7246503900005501  p25=0.715315074800358  p75=0.7341370604557832
tippy_tap_fraction median=0.40883699633699633
slip_ratio         median=0.24619991453505347
tracking_ratio     median=0.9638907917815216  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.99
terminations={'schedule_complete': 99, 'fall': 1}
facing_error_rms   median=0.5803951713706224  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.8166350539155288 (n=100)
       fwd: cmd 0.25 -> achieved 0.263 m/s | tripod median=0.5983313905331129 (n=100)
     lat_l: cmd 0.22 -> achieved 0.217 m/s | tripod median=0.7231044818217267 (n=100)
      diag: cmd 0.25 -> achieved 0.240 m/s | tripod median=0.7387787739689404 (n=100)
      back: cmd 0.22 -> achieved 0.231 m/s | tripod median=0.7024151098826921 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.003 m/s | tripod median=0.7952914272882337 (n=100)
  turn_r90: cmd 0.00 -> achieved -0.000 m/s | tripod median=0.79402256858035 (n=99)
  move_turn: cmd 0.22 -> achieved 0.173 m/s | tripod median=0.6003957790314198 (n=99)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.021 initial=0.007 final=0.021 max=0.050 closed=n/a | settle=0.000 (n=100/100) | progress=0.004 net=-0.000 yaw_rate_ratio=0.483
       fwd: err rms=0.047 initial=0.021 final=0.047 max=0.097 closed=n/a | settle=0.000 (n=100/100) | progress=-0.001 net=0.003 yaw_rate_ratio=0.285
     lat_l: err rms=0.039 initial=0.047 final=0.032 max=0.081 closed=n/a | settle=0.000 (n=100/100) | progress=0.009 net=0.004 yaw_rate_ratio=0.419
      diag: err rms=0.041 initial=0.031 final=0.029 max=0.081 closed=n/a | settle=0.000 (n=100/100) | progress=0.004 net=0.003 yaw_rate_ratio=0.264
      back: err rms=0.038 initial=0.028 final=0.029 max=0.079 closed=n/a | settle=0.000 (n=100/100) | progress=0.008 net=0.005 yaw_rate_ratio=1.864
  turn_l90: err rms=0.840 initial=1.571 final=0.090 max=1.580 closed=0.944 | settle=9.560 (n=95/100) | progress=0.151 net=0.124 yaw_rate_ratio=0.585
  turn_r90: err rms=0.753 initial=1.483 final=0.108 max=1.491 closed=0.928 | settle=10.440 (n=94/99) | progress=0.138 net=0.115 yaw_rate_ratio=0.588
  move_turn: err rms=1.074 initial=1.462 final=0.727 max=1.465 closed=0.498 | settle=n/a (n=0/99) | progress=0.091 net=0.092 yaw_rate_ratio=0.392

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
