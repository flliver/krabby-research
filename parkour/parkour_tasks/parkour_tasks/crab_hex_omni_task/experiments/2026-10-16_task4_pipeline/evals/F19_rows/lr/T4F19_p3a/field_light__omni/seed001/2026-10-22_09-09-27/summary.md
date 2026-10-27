```
=== crab-hex gait eval ===
scenario   : field_light__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_03-27-53_seed19_X07_p3a_A15pB/model_22993.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7723967824040169  p25=0.7632404962836304  p75=0.7795226130778017
tippy_tap_fraction median=0.4106675637287882
slip_ratio         median=0.2437782517592656
tracking_ratio     median=0.932087736542192  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}
facing_error_rms   median=0.595901198811587  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.8400355223189904 (n=100)
       fwd: cmd 0.25 -> achieved 0.252 m/s | tripod median=0.7704775346206907 (n=100)
     lat_l: cmd 0.22 -> achieved 0.205 m/s | tripod median=0.7315466038140703 (n=100)
      diag: cmd 0.25 -> achieved 0.234 m/s | tripod median=0.7027831340146551 (n=100)
      back: cmd 0.22 -> achieved 0.227 m/s | tripod median=0.741039466240935 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.009 m/s | tripod median=0.8199253467796008 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.8100610107381174 (n=100)
  move_turn: cmd 0.22 -> achieved 0.164 m/s | tripod median=0.7322519844299142 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.020 initial=0.011 final=0.018 max=0.049 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.059 initial=0.019 final=0.111 max=0.128 closed=n/a | settle=0.000 (n=96/100) | progress=-0.009 net=0.000 yaw_rate_ratio=0.292
     lat_l: err rms=0.065 initial=0.111 final=0.020 max=0.131 closed=0.795 | settle=0.000 (n=100/100) | progress=0.006 net=0.011 yaw_rate_ratio=0.370
      diag: err rms=0.031 initial=0.020 final=0.028 max=0.073 closed=n/a | settle=0.000 (n=100/100) | progress=-0.006 net=0.002 yaw_rate_ratio=0.213
      back: err rms=0.065 initial=0.027 final=0.101 max=0.131 closed=n/a | settle=0.000 (n=95/100) | progress=-0.003 net=0.013 yaw_rate_ratio=0.379
  turn_l90: err rms=0.969 initial=1.672 final=0.124 max=1.697 closed=0.926 | settle=11.420 (n=78/100) | progress=0.137 net=0.129 yaw_rate_ratio=0.592
  turn_r90: err rms=0.717 initial=1.449 final=0.114 max=1.459 closed=0.921 | settle=10.960 (n=95/100) | progress=0.130 net=0.111 yaw_rate_ratio=0.589
  move_turn: err rms=1.032 initial=1.459 final=0.613 max=1.467 closed=0.582 | settle=n/a (n=0/100) | progress=0.106 net=0.107 yaw_rate_ratio=0.425

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
