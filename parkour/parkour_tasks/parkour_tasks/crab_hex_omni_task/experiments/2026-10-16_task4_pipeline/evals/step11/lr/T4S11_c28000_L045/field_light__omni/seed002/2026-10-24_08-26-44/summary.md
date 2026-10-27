```
=== crab-hex gait eval ===
scenario   : field_light__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_28000.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6944585535601109  p25=0.681700728045314  p75=0.7038515030671376
tippy_tap_fraction median=0.3531301567656766
slip_ratio         median=0.2411866180022291
tracking_ratio     median=0.9497452716024035  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.98
terminations={'schedule_complete': 98, 'fall': 2}
facing_error_rms   median=0.5948167132177767  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.7705436102133197 (n=100)
       fwd: cmd 0.25 -> achieved 0.247 m/s | tripod median=0.5792483562615158 (n=100)
     lat_l: cmd 0.22 -> achieved 0.229 m/s | tripod median=0.6568979185309601 (n=100)
      diag: cmd 0.25 -> achieved 0.252 m/s | tripod median=0.6636712799984674 (n=100)
      back: cmd 0.22 -> achieved 0.217 m/s | tripod median=0.6843587438276418 (n=99)
  turn_l90: cmd 0.00 -> achieved -0.011 m/s | tripod median=0.7778186738756476 (n=99)
  turn_r90: cmd 0.00 -> achieved 0.009 m/s | tripod median=0.7871249283191625 (n=99)
  move_turn: cmd 0.22 -> achieved 0.163 m/s | tripod median=0.5703515108861434 (n=99)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.023 initial=0.007 final=0.025 max=0.050 closed=n/a | settle=0.000 (n=100/100) | progress=-0.013 net=-0.001 yaw_rate_ratio=0.518
       fwd: err rms=0.050 initial=0.025 final=0.034 max=0.100 closed=n/a | settle=0.000 (n=99/100) | progress=-0.001 net=0.005 yaw_rate_ratio=0.396
     lat_l: err rms=0.031 initial=0.034 final=0.024 max=0.076 closed=0.671 | settle=0.000 (n=100/100) | progress=0.001 net=0.002 yaw_rate_ratio=0.507
      diag: err rms=0.041 initial=0.024 final=0.047 max=0.085 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.000 yaw_rate_ratio=n/a
      back: err rms=0.048 initial=0.045 final=0.075 max=0.108 closed=n/a | settle=0.000 (n=98/100) | progress=-0.003 net=0.015 yaw_rate_ratio=0.554
  turn_l90: err rms=0.766 initial=1.496 final=0.019 max=1.511 closed=0.988 | settle=8.040 (n=99/99) | progress=0.167 net=0.125 yaw_rate_ratio=0.642
  turn_r90: err rms=0.787 initial=1.562 final=0.034 max=1.570 closed=0.978 | settle=8.300 (n=99/99) | progress=0.171 net=0.128 yaw_rate_ratio=0.645
  move_turn: err rms=1.184 initial=1.538 final=0.902 max=1.541 closed=0.410 | settle=n/a (n=0/99) | progress=0.077 net=0.079 yaw_rate_ratio=0.425

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
