```
=== crab-hex gait eval ===
scenario   : field_light__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6977948588606  p25=0.6874916440395799  p75=0.706834564676124
tippy_tap_fraction median=0.383458204390394
slip_ratio         median=0.25405008973008375
tracking_ratio     median=0.9366928177173419  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.99
terminations={'schedule_complete': 99, 'fall': 1}
facing_error_rms   median=0.598909038949986  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.7778366924589393 (n=100)
       fwd: cmd 0.25 -> achieved 0.246 m/s | tripod median=0.5765169710420857 (n=100)
     lat_l: cmd 0.22 -> achieved 0.228 m/s | tripod median=0.659556124461252 (n=100)
      diag: cmd 0.25 -> achieved 0.248 m/s | tripod median=0.6507107943808965 (n=100)
      back: cmd 0.22 -> achieved 0.214 m/s | tripod median=0.6750439910314956 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.010 m/s | tripod median=0.7820415869740402 (n=99)
  turn_r90: cmd 0.00 -> achieved 0.012 m/s | tripod median=0.7881343976784111 (n=99)
  move_turn: cmd 0.22 -> achieved 0.161 m/s | tripod median=0.5853058287537622 (n=99)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.023 initial=0.009 final=0.022 max=0.050 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.054 initial=0.023 final=0.050 max=0.117 closed=n/a | settle=0.000 (n=96/100) | progress=-0.003 net=0.007 yaw_rate_ratio=0.389
     lat_l: err rms=0.042 initial=0.051 final=0.028 max=0.093 closed=0.858 | settle=0.000 (n=100/100) | progress=0.002 net=0.004 yaw_rate_ratio=0.588
      diag: err rms=0.041 initial=0.027 final=0.050 max=0.087 closed=n/a | settle=0.000 (n=100/100) | progress=-0.003 net=0.001 yaw_rate_ratio=0.432
      back: err rms=0.056 initial=0.050 final=0.076 max=0.120 closed=n/a | settle=0.000 (n=97/100) | progress=-0.003 net=0.015 yaw_rate_ratio=0.673
  turn_l90: err rms=0.768 initial=1.495 final=0.023 max=1.503 closed=0.984 | settle=8.000 (n=99/99) | progress=0.170 net=0.124 yaw_rate_ratio=0.641
  turn_r90: err rms=0.784 initial=1.561 final=0.043 max=1.564 closed=0.973 | settle=8.180 (n=99/99) | progress=0.173 net=0.127 yaw_rate_ratio=0.644
  move_turn: err rms=1.225 initial=1.529 final=0.954 max=1.531 closed=0.381 | settle=n/a (n=0/99) | progress=0.072 net=0.073 yaw_rate_ratio=0.423

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
