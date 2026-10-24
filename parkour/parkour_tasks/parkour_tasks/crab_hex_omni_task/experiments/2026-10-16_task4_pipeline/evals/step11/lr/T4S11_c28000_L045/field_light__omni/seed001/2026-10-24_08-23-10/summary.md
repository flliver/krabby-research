```
=== crab-hex gait eval ===
scenario   : field_light__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_28000.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6961570065937708  p25=0.6824850880810924  p75=0.7099550775767527
tippy_tap_fraction median=0.35151665426894785
slip_ratio         median=0.23327565394292088
tracking_ratio     median=0.949289844772329  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}
facing_error_rms   median=0.5979855541152805  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.7726518535419329 (n=100)
       fwd: cmd 0.25 -> achieved 0.245 m/s | tripod median=0.5761348660320853 (n=100)
     lat_l: cmd 0.22 -> achieved 0.229 m/s | tripod median=0.6599049957535761 (n=100)
      diag: cmd 0.25 -> achieved 0.251 m/s | tripod median=0.6607841884874643 (n=100)
      back: cmd 0.22 -> achieved 0.219 m/s | tripod median=0.6781703588845953 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.012 m/s | tripod median=0.778224512281775 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.009 m/s | tripod median=0.7896396671885997 (n=100)
  move_turn: cmd 0.22 -> achieved 0.161 m/s | tripod median=0.5835082224515895 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.023 initial=0.008 final=0.029 max=0.050 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.054 initial=0.029 final=0.037 max=0.111 closed=n/a | settle=0.000 (n=99/100) | progress=-0.002 net=0.006 yaw_rate_ratio=0.374
     lat_l: err rms=0.040 initial=0.037 final=0.029 max=0.079 closed=0.533 | settle=0.000 (n=100/100) | progress=-0.004 net=0.004 yaw_rate_ratio=0.387
      diag: err rms=0.041 initial=0.029 final=0.048 max=0.083 closed=n/a | settle=0.000 (n=100/100) | progress=-0.002 net=-0.001 yaw_rate_ratio=0.517
      back: err rms=0.053 initial=0.049 final=0.077 max=0.109 closed=n/a | settle=0.000 (n=95/100) | progress=-0.010 net=0.015 yaw_rate_ratio=0.680
  turn_l90: err rms=0.768 initial=1.500 final=0.021 max=1.508 closed=0.985 | settle=8.090 (n=100/100) | progress=0.167 net=0.125 yaw_rate_ratio=0.643
  turn_r90: err rms=0.787 initial=1.565 final=0.035 max=1.577 closed=0.978 | settle=8.240 (n=100/100) | progress=0.174 net=0.128 yaw_rate_ratio=0.646
  move_turn: err rms=1.209 initial=1.546 final=0.932 max=1.550 closed=0.399 | settle=n/a (n=0/100) | progress=0.076 net=0.077 yaw_rate_ratio=0.418

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
