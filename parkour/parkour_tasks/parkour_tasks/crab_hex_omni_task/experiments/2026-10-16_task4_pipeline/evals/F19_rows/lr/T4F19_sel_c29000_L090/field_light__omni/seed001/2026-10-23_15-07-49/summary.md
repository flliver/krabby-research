```
=== crab-hex gait eval ===
scenario   : field_light__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_29000.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6761990706816978  p25=0.6641588894798496  p75=0.6870813268326981
tippy_tap_fraction median=0.32450331125827814
slip_ratio         median=0.23798942011830734
tracking_ratio     median=0.959967109372284  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.96
terminations={'schedule_complete': 96, 'fall': 4}
facing_error_rms   median=0.6040121124622091  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7683100992043643 (n=100)
       fwd: cmd 0.25 -> achieved 0.247 m/s | tripod median=0.5384889905866816 (n=100)
     lat_l: cmd 0.22 -> achieved 0.229 m/s | tripod median=0.6339175498521126 (n=100)
      diag: cmd 0.25 -> achieved 0.254 m/s | tripod median=0.635607920828111 (n=100)
      back: cmd 0.22 -> achieved 0.220 m/s | tripod median=0.6598312511643016 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.010 m/s | tripod median=0.7726283952102497 (n=99)
  turn_r90: cmd 0.00 -> achieved 0.009 m/s | tripod median=0.7790118778937356 (n=99)
  move_turn: cmd 0.22 -> achieved 0.172 m/s | tripod median=0.5255120755181306 (n=99)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.017 initial=0.007 final=0.022 max=0.040 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.060 initial=0.022 final=0.045 max=0.115 closed=n/a | settle=0.000 (n=95/100) | progress=0.000 net=0.003 yaw_rate_ratio=0.496
     lat_l: err rms=0.033 initial=0.042 final=0.034 max=0.073 closed=0.708 | settle=0.000 (n=100/100) | progress=-0.002 net=0.005 yaw_rate_ratio=0.567
      diag: err rms=0.041 initial=0.034 final=0.052 max=0.082 closed=n/a | settle=0.000 (n=100/100) | progress=-0.003 net=-0.001 yaw_rate_ratio=0.460
      back: err rms=0.050 initial=0.052 final=0.058 max=0.110 closed=n/a | settle=0.000 (n=98/100) | progress=-0.004 net=0.014 yaw_rate_ratio=0.496
  turn_l90: err rms=0.769 initial=1.513 final=0.020 max=1.525 closed=0.987 | settle=8.000 (n=99/99) | progress=0.171 net=0.127 yaw_rate_ratio=0.637
  turn_r90: err rms=0.800 initial=1.576 final=0.044 max=1.580 closed=0.971 | settle=8.480 (n=99/99) | progress=0.170 net=0.127 yaw_rate_ratio=0.641
  move_turn: err rms=1.222 initial=1.528 final=0.967 max=1.536 closed=0.372 | settle=n/a (n=0/99) | progress=0.070 net=0.072 yaw_rate_ratio=0.431

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
