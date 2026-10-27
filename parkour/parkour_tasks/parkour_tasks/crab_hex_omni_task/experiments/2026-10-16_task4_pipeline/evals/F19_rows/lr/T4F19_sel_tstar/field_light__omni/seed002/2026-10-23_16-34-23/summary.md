```
=== crab-hex gait eval ===
scenario   : field_light__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7046160656450042  p25=0.6912920459378438  p75=0.7144448156198638
tippy_tap_fraction median=0.42441171491450824
slip_ratio         median=0.2560791827693968
tracking_ratio     median=0.9268153355161837  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}
facing_error_rms   median=0.5927951797377446  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.7796119222216145 (n=100)
       fwd: cmd 0.25 -> achieved 0.242 m/s | tripod median=0.5868256791435498 (n=100)
     lat_l: cmd 0.22 -> achieved 0.226 m/s | tripod median=0.6564140209683282 (n=100)
      diag: cmd 0.25 -> achieved 0.246 m/s | tripod median=0.6620456568674531 (n=100)
      back: cmd 0.22 -> achieved 0.210 m/s | tripod median=0.6923040163472463 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.008 m/s | tripod median=0.7856162477181194 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.011 m/s | tripod median=0.79169166837838 (n=100)
  move_turn: cmd 0.22 -> achieved 0.156 m/s | tripod median=0.6141666926834373 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.022 initial=0.008 final=0.026 max=0.049 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.000 yaw_rate_ratio=n/a
       fwd: err rms=0.064 initial=0.027 final=0.066 max=0.129 closed=n/a | settle=0.000 (n=96/100) | progress=-0.001 net=0.009 yaw_rate_ratio=0.415
     lat_l: err rms=0.050 initial=0.066 final=0.035 max=0.102 closed=0.584 | settle=0.000 (n=100/100) | progress=0.001 net=0.005 yaw_rate_ratio=0.545
      diag: err rms=0.044 initial=0.034 final=0.041 max=0.087 closed=n/a | settle=0.000 (n=100/100) | progress=-0.001 net=0.002 yaw_rate_ratio=0.378
      back: err rms=0.055 initial=0.041 final=0.089 max=0.117 closed=n/a | settle=0.000 (n=97/100) | progress=-0.002 net=0.016 yaw_rate_ratio=0.517
  turn_l90: err rms=0.749 initial=1.481 final=0.023 max=1.488 closed=0.985 | settle=7.920 (n=100/100) | progress=0.170 net=0.123 yaw_rate_ratio=0.648
  turn_r90: err rms=0.774 initial=1.559 final=0.037 max=1.562 closed=0.977 | settle=8.010 (n=100/100) | progress=0.176 net=0.127 yaw_rate_ratio=0.653
  move_turn: err rms=1.207 initial=1.535 final=0.930 max=1.545 closed=0.392 | settle=n/a (n=0/100) | progress=0.075 net=0.076 yaw_rate_ratio=0.409

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
