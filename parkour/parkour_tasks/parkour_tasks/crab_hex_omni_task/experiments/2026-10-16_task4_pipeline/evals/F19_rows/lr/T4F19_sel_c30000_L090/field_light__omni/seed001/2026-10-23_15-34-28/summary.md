```
=== crab-hex gait eval ===
scenario   : field_light__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_30000.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6478760425354979  p25=0.6288801904077571  p75=0.661873623243784
tippy_tap_fraction median=0.3333333333333333
slip_ratio         median=0.23065224464352824
tracking_ratio     median=0.9267614282634193  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.83
terminations={'schedule_complete': 83, 'fall': 17}
facing_error_rms   median=0.5839372821123201  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved -0.001 m/s | tripod median=0.7635217354534537 (n=100)
       fwd: cmd 0.25 -> achieved 0.244 m/s | tripod median=0.549890023190094 (n=100)
     lat_l: cmd 0.22 -> achieved 0.226 m/s | tripod median=0.5805747725182552 (n=99)
      diag: cmd 0.25 -> achieved 0.251 m/s | tripod median=0.5529956551100446 (n=99)
      back: cmd 0.22 -> achieved 0.218 m/s | tripod median=0.733899517442383 (n=95)
  turn_l90: cmd 0.00 -> achieved -0.009 m/s | tripod median=0.762258560511311 (n=95)
  turn_r90: cmd 0.00 -> achieved 0.005 m/s | tripod median=0.7753561115349236 (n=95)
  move_turn: cmd 0.22 -> achieved 0.155 m/s | tripod median=0.36288132481486113 (n=95)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.029 initial=0.016 final=0.035 max=0.051 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.002 yaw_rate_ratio=n/a
       fwd: err rms=0.045 initial=0.035 final=0.030 max=0.114 closed=n/a | settle=0.000 (n=99/100) | progress=-0.002 net=0.003 yaw_rate_ratio=0.752
     lat_l: err rms=0.031 initial=0.032 final=0.028 max=0.068 closed=n/a | settle=0.000 (n=99/99) | progress=0.002 net=0.001 yaw_rate_ratio=0.442
      diag: err rms=0.032 initial=0.029 final=0.025 max=0.071 closed=n/a | settle=0.000 (n=99/99) | progress=0.000 net=0.003 yaw_rate_ratio=0.558
      back: err rms=0.078 initial=0.025 final=0.116 max=0.147 closed=n/a | settle=0.000 (n=91/95) | progress=-0.002 net=0.015 yaw_rate_ratio=0.308
  turn_l90: err rms=0.734 initial=1.456 final=0.025 max=1.473 closed=0.984 | settle=7.840 (n=95/95) | progress=0.168 net=0.123 yaw_rate_ratio=0.616
  turn_r90: err rms=0.787 initial=1.590 final=0.075 max=1.593 closed=0.951 | settle=8.200 (n=95/95) | progress=0.175 net=0.126 yaw_rate_ratio=0.647
  move_turn: err rms=1.192 initial=1.496 final=0.999 max=1.502 closed=0.336 | settle=n/a (n=0/95) | progress=0.063 net=0.065 yaw_rate_ratio=0.434

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
