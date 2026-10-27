```
=== crab-hex gait eval ===
scenario   : ring_obst__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7035652529990264  p25=0.6918528691772434  p75=0.7123386961109579
tippy_tap_fraction median=0.4224658632238857
slip_ratio         median=0.27098278595976477
tracking_ratio     median=0.9279685226548646  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}
facing_error_rms   median=0.5933111070875015  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.7882883659490749 (n=100)
       fwd: cmd 0.25 -> achieved 0.242 m/s | tripod median=0.5843653604915375 (n=100)
     lat_l: cmd 0.22 -> achieved 0.226 m/s | tripod median=0.6431542562721232 (n=100)
      diag: cmd 0.25 -> achieved 0.247 m/s | tripod median=0.6566748255346079 (n=100)
      back: cmd 0.22 -> achieved 0.213 m/s | tripod median=0.6986037685511494 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.007 m/s | tripod median=0.7746715080158418 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.011 m/s | tripod median=0.7911469359480571 (n=100)
  move_turn: cmd 0.22 -> achieved 0.158 m/s | tripod median=0.6307573916285945 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.019 initial=0.010 final=0.021 max=0.043 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a
       fwd: err rms=0.063 initial=0.021 final=0.065 max=0.118 closed=n/a | settle=0.000 (n=99/100) | progress=-0.002 net=0.009 yaw_rate_ratio=0.441
     lat_l: err rms=0.052 initial=0.065 final=0.030 max=0.108 closed=0.589 | settle=0.000 (n=100/100) | progress=-0.001 net=0.004 yaw_rate_ratio=0.401
      diag: err rms=0.041 initial=0.032 final=0.038 max=0.085 closed=n/a | settle=0.000 (n=100/100) | progress=0.004 net=-0.000 yaw_rate_ratio=0.501
      back: err rms=0.057 initial=0.035 final=0.088 max=0.118 closed=n/a | settle=0.000 (n=95/100) | progress=-0.007 net=0.014 yaw_rate_ratio=0.634
  turn_l90: err rms=0.750 initial=1.484 final=0.021 max=1.491 closed=0.986 | settle=7.820 (n=100/100) | progress=0.171 net=0.123 yaw_rate_ratio=0.649
  turn_r90: err rms=0.781 initial=1.567 final=0.038 max=1.570 closed=0.976 | settle=8.070 (n=100/100) | progress=0.175 net=0.127 yaw_rate_ratio=0.656
  move_turn: err rms=1.203 initial=1.533 final=0.912 max=1.538 closed=0.407 | settle=n/a (n=0/100) | progress=0.077 net=0.078 yaw_rate_ratio=0.415

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
