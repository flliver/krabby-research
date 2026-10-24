```
=== crab-hex gait eval ===
scenario   : field_hard__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6551980253518337  p25=0.6117215823356555  p75=0.6800385679634325
tippy_tap_fraction median=0.42909197766676466
slip_ratio         median=0.27084926118404046
tracking_ratio     median=0.8851720971248157  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.88
terminations={'schedule_complete': 88, 'fall': 12}
facing_error_rms   median=0.5962167116291139  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.7912364634967988 (n=100)
       fwd: cmd 0.25 -> achieved 0.241 m/s | tripod median=0.5845558636643503 (n=100)
     lat_l: cmd 0.22 -> achieved 0.218 m/s | tripod median=0.6047968404205559 (n=100)
      diag: cmd 0.25 -> achieved 0.234 m/s | tripod median=0.5708786027696897 (n=97)
      back: cmd 0.22 -> achieved 0.209 m/s | tripod median=0.6381995534407567 (n=91)
  turn_l90: cmd 0.00 -> achieved -0.008 m/s | tripod median=0.7464296367946467 (n=89)
  turn_r90: cmd 0.00 -> achieved 0.010 m/s | tripod median=0.7544728697185843 (n=89)
  move_turn: cmd 0.22 -> achieved 0.148 m/s | tripod median=0.5701830872843614 (n=89)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.019 initial=0.008 final=0.016 max=0.045 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.058 initial=0.016 final=0.060 max=0.116 closed=n/a | settle=0.000 (n=95/100) | progress=0.000 net=0.007 yaw_rate_ratio=0.435
     lat_l: err rms=0.047 initial=0.058 final=0.027 max=0.096 closed=0.518 | settle=0.000 (n=100/100) | progress=0.002 net=0.005 yaw_rate_ratio=0.486
      diag: err rms=0.055 initial=0.028 final=0.051 max=0.108 closed=n/a | settle=0.000 (n=90/100) | progress=-0.004 net=0.003 yaw_rate_ratio=0.456
      back: err rms=0.071 initial=0.050 final=0.097 max=0.132 closed=0.651 | settle=0.000 (n=84/91) | progress=-0.001 net=0.011 yaw_rate_ratio=0.478
  turn_l90: err rms=0.755 initial=1.476 final=0.026 max=1.482 closed=0.983 | settle=8.080 (n=89/89) | progress=0.166 net=0.123 yaw_rate_ratio=0.636
  turn_r90: err rms=0.789 initial=1.570 final=0.040 max=1.571 closed=0.975 | settle=8.480 (n=89/89) | progress=0.168 net=0.127 yaw_rate_ratio=0.639
  move_turn: err rms=1.211 initial=1.532 final=0.942 max=1.538 closed=0.391 | settle=n/a (n=0/89) | progress=0.074 net=0.075 yaw_rate_ratio=0.418

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
