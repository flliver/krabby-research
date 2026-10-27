```
=== crab-hex gait eval ===
scenario   : field_light__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6599836775541494  p25=0.6485939271134188  p75=0.6723280692032885
tippy_tap_fraction median=0.3586103623839473
slip_ratio         median=0.23595590093450508
tracking_ratio     median=0.9579922907195577  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.87
terminations={'schedule_complete': 87, 'fall': 13}
facing_error_rms   median=0.6091746518598706  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.7469332346696632 (n=100)
       fwd: cmd 0.25 -> achieved 0.250 m/s | tripod median=0.5814865905726951 (n=100)
     lat_l: cmd 0.22 -> achieved 0.223 m/s | tripod median=0.6593719256921188 (n=100)
      diag: cmd 0.25 -> achieved 0.249 m/s | tripod median=0.6221477277814252 (n=100)
      back: cmd 0.22 -> achieved 0.220 m/s | tripod median=0.6277580642321863 (n=98)
  turn_l90: cmd 0.00 -> achieved -0.011 m/s | tripod median=0.7606145863725148 (n=96)
  turn_r90: cmd 0.00 -> achieved 0.011 m/s | tripod median=0.7579179782677132 (n=96)
  move_turn: cmd 0.22 -> achieved 0.179 m/s | tripod median=0.5014785441479108 (n=96)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.017 initial=0.007 final=0.017 max=0.043 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.057 initial=0.017 final=0.040 max=0.116 closed=n/a | settle=0.000 (n=94/100) | progress=-0.001 net=0.000 yaw_rate_ratio=0.534
     lat_l: err rms=0.029 initial=0.039 final=0.018 max=0.067 closed=0.900 | settle=0.000 (n=100/100) | progress=0.060 net=0.005 yaw_rate_ratio=0.655
      diag: err rms=0.039 initial=0.017 final=0.033 max=0.082 closed=n/a | settle=0.000 (n=97/100) | progress=-0.008 net=0.000 yaw_rate_ratio=0.474
      back: err rms=0.052 initial=0.034 final=0.071 max=0.110 closed=0.411 | settle=0.000 (n=94/98) | progress=-0.003 net=0.011 yaw_rate_ratio=0.619
  turn_l90: err rms=0.784 initial=1.505 final=0.025 max=1.521 closed=0.984 | settle=8.230 (n=96/96) | progress=0.166 net=0.124 yaw_rate_ratio=0.623
  turn_r90: err rms=0.788 initial=1.563 final=0.040 max=1.566 closed=0.974 | settle=8.320 (n=95/96) | progress=0.171 net=0.126 yaw_rate_ratio=0.640
  move_turn: err rms=1.241 initial=1.533 final=1.019 max=1.537 closed=0.331 | settle=n/a (n=0/96) | progress=0.064 net=0.065 yaw_rate_ratio=0.439

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
