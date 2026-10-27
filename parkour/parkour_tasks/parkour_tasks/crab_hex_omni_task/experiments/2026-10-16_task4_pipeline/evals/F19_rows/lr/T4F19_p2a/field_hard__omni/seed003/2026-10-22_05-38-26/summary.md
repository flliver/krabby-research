```
=== crab-hex gait eval ===
scenario   : field_hard__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-21_23-06-12_seed19_X06_p2a_A15pB/model_19994.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6564620394927292  p25=0.6233458826541823  p75=0.684929467596732
tippy_tap_fraction median=0.40563396057347667
slip_ratio         median=0.25799333515350964
tracking_ratio     median=0.9203325912032472  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.78
terminations={'fall': 22, 'schedule_complete': 78}
facing_error_rms   median=0.5880387923944816  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.8153984029429509 (n=100)
       fwd: cmd 0.25 -> achieved 0.259 m/s | tripod median=0.5860095664438955 (n=100)
     lat_l: cmd 0.22 -> achieved 0.206 m/s | tripod median=0.6618908831129927 (n=100)
      diag: cmd 0.25 -> achieved 0.231 m/s | tripod median=0.6311302800190264 (n=96)
      back: cmd 0.22 -> achieved 0.226 m/s | tripod median=0.6346239500678845 (n=91)
  turn_l90: cmd 0.00 -> achieved -0.002 m/s | tripod median=0.739799677635709 (n=85)
  turn_r90: cmd 0.00 -> achieved -0.000 m/s | tripod median=0.7319022416554193 (n=83)
  move_turn: cmd 0.22 -> achieved 0.163 m/s | tripod median=0.5200935847229493 (n=83)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.022 initial=0.006 final=0.025 max=0.050 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.000 yaw_rate_ratio=n/a
       fwd: err rms=0.050 initial=0.025 final=0.050 max=0.094 closed=n/a | settle=0.000 (n=98/100) | progress=-0.002 net=0.000 yaw_rate_ratio=0.367
     lat_l: err rms=0.045 initial=0.050 final=0.035 max=0.091 closed=0.506 | settle=0.000 (n=100/100) | progress=0.001 net=0.004 yaw_rate_ratio=0.430
      diag: err rms=0.054 initial=0.034 final=0.051 max=0.103 closed=n/a | settle=0.000 (n=92/98) | progress=-0.001 net=-0.001 yaw_rate_ratio=0.471
      back: err rms=0.052 initial=0.049 final=0.032 max=0.113 closed=0.796 | settle=0.000 (n=86/92) | progress=0.000 net=0.008 yaw_rate_ratio=0.556
  turn_l90: err rms=0.864 initial=1.585 final=0.088 max=1.595 closed=0.944 | settle=9.820 (n=76/86) | progress=0.148 net=0.125 yaw_rate_ratio=0.576
  turn_r90: err rms=0.762 initial=1.484 final=0.111 max=1.490 closed=0.923 | settle=10.400 (n=70/84) | progress=0.133 net=0.114 yaw_rate_ratio=0.582
  move_turn: err rms=1.081 initial=1.462 final=0.765 max=1.464 closed=0.466 | settle=n/a (n=0/83) | progress=0.089 net=0.089 yaw_rate_ratio=0.400

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
