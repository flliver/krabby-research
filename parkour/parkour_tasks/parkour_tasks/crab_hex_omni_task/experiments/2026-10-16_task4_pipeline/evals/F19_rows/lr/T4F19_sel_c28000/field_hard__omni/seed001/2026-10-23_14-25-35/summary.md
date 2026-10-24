```
=== crab-hex gait eval ===
scenario   : field_hard__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_28000.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.657044760444083  p25=0.6237904692707162  p75=0.6812862223964253
tippy_tap_fraction median=0.39014538433143087
slip_ratio         median=0.26378817798829257
tracking_ratio     median=0.8818170629397013  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.9
terminations={'schedule_complete': 90, 'fall': 10}
facing_error_rms   median=0.5954956864641548  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.7837163728308336 (n=100)
       fwd: cmd 0.25 -> achieved 0.238 m/s | tripod median=0.5761366717373613 (n=100)
     lat_l: cmd 0.22 -> achieved 0.219 m/s | tripod median=0.6012456026316504 (n=100)
      diag: cmd 0.25 -> achieved 0.231 m/s | tripod median=0.5683905782312046 (n=99)
      back: cmd 0.22 -> achieved 0.211 m/s | tripod median=0.6328123391753492 (n=91)
  turn_l90: cmd 0.00 -> achieved -0.011 m/s | tripod median=0.7589161260886349 (n=90)
  turn_r90: cmd 0.00 -> achieved 0.007 m/s | tripod median=0.7708814589587185 (n=90)
  move_turn: cmd 0.22 -> achieved 0.145 m/s | tripod median=0.5636547602392004 (n=90)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.021 initial=0.008 final=0.025 max=0.046 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.058 initial=0.025 final=0.057 max=0.123 closed=n/a | settle=0.000 (n=99/100) | progress=-0.002 net=0.009 yaw_rate_ratio=0.391
     lat_l: err rms=0.043 initial=0.059 final=0.028 max=0.091 closed=0.696 | settle=0.000 (n=99/100) | progress=-0.000 net=0.005 yaw_rate_ratio=0.529
      diag: err rms=0.048 initial=0.027 final=0.043 max=0.093 closed=-0.305 | settle=0.000 (n=87/100) | progress=-0.026 net=0.004 yaw_rate_ratio=0.458
      back: err rms=0.069 initial=0.041 final=0.090 max=0.138 closed=0.587 | settle=0.000 (n=85/92) | progress=0.000 net=0.012 yaw_rate_ratio=0.508
  turn_l90: err rms=0.770 initial=1.484 final=0.029 max=1.485 closed=0.981 | settle=8.300 (n=88/90) | progress=0.161 net=0.123 yaw_rate_ratio=0.627
  turn_r90: err rms=0.779 initial=1.556 final=0.036 max=1.558 closed=0.977 | settle=8.320 (n=89/90) | progress=0.169 net=0.127 yaw_rate_ratio=0.641
  move_turn: err rms=1.206 initial=1.538 final=0.929 max=1.542 closed=0.395 | settle=n/a (n=0/90) | progress=0.075 net=0.076 yaw_rate_ratio=0.421

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
