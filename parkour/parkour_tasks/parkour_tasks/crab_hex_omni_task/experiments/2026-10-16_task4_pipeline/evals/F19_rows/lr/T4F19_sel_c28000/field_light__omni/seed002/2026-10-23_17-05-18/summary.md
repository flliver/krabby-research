```
=== crab-hex gait eval ===
scenario   : field_light__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_28000.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7061522823440498  p25=0.6959810141729199  p75=0.7127105824052502
tippy_tap_fraction median=0.386235057403243
slip_ratio         median=0.256571971686542
tracking_ratio     median=0.934642138003389  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.99
terminations={'schedule_complete': 99, 'fall': 1}
facing_error_rms   median=0.5912644320659901  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.7724569597305166 (n=100)
       fwd: cmd 0.25 -> achieved 0.243 m/s | tripod median=0.5717596997468133 (n=100)
     lat_l: cmd 0.22 -> achieved 0.228 m/s | tripod median=0.663108669196973 (n=100)
      diag: cmd 0.25 -> achieved 0.251 m/s | tripod median=0.6721532409332795 (n=100)
      back: cmd 0.22 -> achieved 0.214 m/s | tripod median=0.6952046981530435 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.011 m/s | tripod median=0.7840505651554909 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.010 m/s | tripod median=0.7992671207586903 (n=100)
  move_turn: cmd 0.22 -> achieved 0.154 m/s | tripod median=0.6189621062377471 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.024 initial=0.008 final=0.030 max=0.050 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.000 yaw_rate_ratio=n/a
       fwd: err rms=0.059 initial=0.032 final=0.046 max=0.111 closed=n/a | settle=0.000 (n=99/100) | progress=0.000 net=0.009 yaw_rate_ratio=0.410
     lat_l: err rms=0.040 initial=0.046 final=0.031 max=0.089 closed=0.328 | settle=0.000 (n=100/100) | progress=-0.002 net=0.004 yaw_rate_ratio=0.533
      diag: err rms=0.043 initial=0.030 final=0.043 max=0.088 closed=n/a | settle=0.000 (n=100/100) | progress=0.001 net=-0.001 yaw_rate_ratio=0.457
      back: err rms=0.054 initial=0.042 final=0.085 max=0.114 closed=n/a | settle=0.000 (n=98/100) | progress=-0.002 net=0.015 yaw_rate_ratio=0.642
  turn_l90: err rms=0.751 initial=1.485 final=0.024 max=1.492 closed=0.984 | settle=7.970 (n=100/100) | progress=0.168 net=0.123 yaw_rate_ratio=0.643
  turn_r90: err rms=0.778 initial=1.564 final=0.032 max=1.568 closed=0.979 | settle=8.140 (n=100/100) | progress=0.175 net=0.127 yaw_rate_ratio=0.650
  move_turn: err rms=1.191 initial=1.540 final=0.897 max=1.545 closed=0.414 | settle=n/a (n=0/100) | progress=0.079 net=0.079 yaw_rate_ratio=0.412

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
