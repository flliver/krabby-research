```
=== crab-hex gait eval ===
scenario   : field_hard__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_30000.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6263272556517154  p25=0.5732766648472611  p75=0.6636062544257766
tippy_tap_fraction median=0.3765432098765432
slip_ratio         median=0.2633976252513893
tracking_ratio     median=0.8572553324709202  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.74
terminations={'schedule_complete': 74, 'fall': 26}
facing_error_rms   median=0.5835124175715409  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved -0.001 m/s | tripod median=0.8092549845281365 (n=100)
       fwd: cmd 0.25 -> achieved 0.234 m/s | tripod median=0.6090180875609847 (n=100)
     lat_l: cmd 0.22 -> achieved 0.210 m/s | tripod median=0.5451836346143782 (n=99)
      diag: cmd 0.25 -> achieved 0.220 m/s | tripod median=0.4568127352773552 (n=94)
      back: cmd 0.22 -> achieved 0.213 m/s | tripod median=0.6850460891334074 (n=79)
  turn_l90: cmd 0.00 -> achieved -0.007 m/s | tripod median=0.7383495899903806 (n=77)
  turn_r90: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.7595154535304471 (n=76)
  move_turn: cmd 0.22 -> achieved 0.138 m/s | tripod median=0.4672392948211987 (n=76)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.029 initial=0.017 final=0.037 max=0.050 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.002 yaw_rate_ratio=n/a
       fwd: err rms=0.073 initial=0.038 final=0.044 max=0.143 closed=n/a | settle=0.000 (n=95/100) | progress=-0.002 net=0.002 yaw_rate_ratio=0.621
     lat_l: err rms=0.046 initial=0.043 final=0.039 max=0.098 closed=0.764 | settle=0.000 (n=99/100) | progress=0.001 net=0.003 yaw_rate_ratio=0.439
      diag: err rms=0.044 initial=0.036 final=0.040 max=0.092 closed=-0.691 | settle=0.000 (n=92/97) | progress=-0.005 net=0.001 yaw_rate_ratio=0.430
      back: err rms=0.080 initial=0.038 final=0.106 max=0.146 closed=0.707 | settle=0.000 (n=75/79) | progress=-0.001 net=0.002 yaw_rate_ratio=0.366
  turn_l90: err rms=0.720 initial=1.466 final=0.027 max=1.470 closed=0.982 | settle=7.720 (n=73/77) | progress=0.171 net=0.123 yaw_rate_ratio=0.625
  turn_r90: err rms=0.795 initial=1.576 final=0.068 max=1.577 closed=0.958 | settle=8.300 (n=74/76) | progress=0.171 net=0.126 yaw_rate_ratio=0.638
  move_turn: err rms=1.192 initial=1.503 final=0.961 max=1.509 closed=0.362 | settle=n/a (n=0/76) | progress=0.067 net=0.068 yaw_rate_ratio=0.418

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
