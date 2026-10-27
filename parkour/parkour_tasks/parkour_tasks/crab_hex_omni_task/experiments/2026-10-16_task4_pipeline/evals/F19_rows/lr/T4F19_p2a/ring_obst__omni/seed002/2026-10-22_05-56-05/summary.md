```
=== crab-hex gait eval ===
scenario   : ring_obst__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-21_23-06-12_seed19_X06_p2a_A15pB/model_19994.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7249609947914826  p25=0.711645388066664  p75=0.7376389168347955
tippy_tap_fraction median=0.4141267648864334
slip_ratio         median=0.24416110540346408
tracking_ratio     median=0.9620464978724217  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}
facing_error_rms   median=0.5857661494339893  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.000 m/s | tripod median=0.8181708616214018 (n=100)
       fwd: cmd 0.25 -> achieved 0.260 m/s | tripod median=0.606220912271713 (n=100)
     lat_l: cmd 0.22 -> achieved 0.216 m/s | tripod median=0.7174928053294376 (n=100)
      diag: cmd 0.25 -> achieved 0.239 m/s | tripod median=0.736481096003489 (n=100)
      back: cmd 0.22 -> achieved 0.231 m/s | tripod median=0.7060251656074599 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.003 m/s | tripod median=0.7871199088049713 (n=100)
  turn_r90: cmd 0.00 -> achieved -0.001 m/s | tripod median=0.7842943974313041 (n=100)
  move_turn: cmd 0.22 -> achieved 0.172 m/s | tripod median=0.6043428424874704 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.020 initial=0.005 final=0.023 max=0.048 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.050 initial=0.023 final=0.039 max=0.097 closed=n/a | settle=0.000 (n=100/100) | progress=-0.001 net=0.003 yaw_rate_ratio=0.277
     lat_l: err rms=0.038 initial=0.038 final=0.029 max=0.076 closed=n/a | settle=0.000 (n=100/100) | progress=0.009 net=0.003 yaw_rate_ratio=0.580
      diag: err rms=0.043 initial=0.030 final=0.031 max=0.083 closed=n/a | settle=0.000 (n=99/100) | progress=0.002 net=0.002 yaw_rate_ratio=0.387
      back: err rms=0.038 initial=0.031 final=0.027 max=0.081 closed=0.879 | settle=0.000 (n=100/100) | progress=-0.001 net=0.005 yaw_rate_ratio=0.538
  turn_l90: err rms=0.860 initial=1.579 final=0.088 max=1.586 closed=0.944 | settle=9.740 (n=100/100) | progress=0.151 net=0.124 yaw_rate_ratio=0.585
  turn_r90: err rms=0.756 initial=1.484 final=0.106 max=1.489 closed=0.929 | settle=10.420 (n=93/100) | progress=0.137 net=0.115 yaw_rate_ratio=0.582
  move_turn: err rms=1.080 initial=1.465 final=0.744 max=1.467 closed=0.490 | settle=n/a (n=0/100) | progress=0.090 net=0.090 yaw_rate_ratio=0.391

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
