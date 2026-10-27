```
=== crab-hex gait eval ===
scenario   : field_light__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7023397734809533  p25=0.6938493317421559  p75=0.7140341099188353
tippy_tap_fraction median=0.42097829319060365
slip_ratio         median=0.27167625718573385
tracking_ratio     median=0.9266473120215084  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}
facing_error_rms   median=0.598870295554832  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.7886584092713912 (n=100)
       fwd: cmd 0.25 -> achieved 0.244 m/s | tripod median=0.5828721842750602 (n=100)
     lat_l: cmd 0.22 -> achieved 0.227 m/s | tripod median=0.6490224051584219 (n=100)
      diag: cmd 0.25 -> achieved 0.246 m/s | tripod median=0.6576553317478103 (n=100)
      back: cmd 0.22 -> achieved 0.211 m/s | tripod median=0.6859037569415865 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.009 m/s | tripod median=0.78587306110472 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.011 m/s | tripod median=0.7911594958356635 (n=100)
  move_turn: cmd 0.22 -> achieved 0.157 m/s | tripod median=0.6237970899729726 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.019 initial=0.009 final=0.023 max=0.044 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.059 initial=0.023 final=0.067 max=0.118 closed=n/a | settle=0.000 (n=95/100) | progress=-0.001 net=0.007 yaw_rate_ratio=0.433
     lat_l: err rms=0.041 initial=0.067 final=0.026 max=0.094 closed=0.667 | settle=0.000 (n=100/100) | progress=0.004 net=0.004 yaw_rate_ratio=0.593
      diag: err rms=0.049 initial=0.027 final=0.042 max=0.093 closed=n/a | settle=0.000 (n=99/100) | progress=0.001 net=-0.000 yaw_rate_ratio=0.465
      back: err rms=0.058 initial=0.042 final=0.081 max=0.120 closed=0.448 | settle=0.000 (n=95/100) | progress=-0.005 net=0.014 yaw_rate_ratio=0.570
  turn_l90: err rms=0.752 initial=1.490 final=0.022 max=1.495 closed=0.985 | settle=7.870 (n=100/100) | progress=0.171 net=0.123 yaw_rate_ratio=0.652
  turn_r90: err rms=0.774 initial=1.559 final=0.032 max=1.562 closed=0.980 | settle=8.220 (n=100/100) | progress=0.172 net=0.128 yaw_rate_ratio=0.651
  move_turn: err rms=1.212 initial=1.541 final=0.932 max=1.548 closed=0.401 | settle=n/a (n=0/100) | progress=0.077 net=0.078 yaw_rate_ratio=0.414

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
