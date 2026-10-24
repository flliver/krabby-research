```
=== crab-hex gait eval ===
scenario   : ring_obst__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7036151338012173  p25=0.6876530340701527  p75=0.7142849333842992
tippy_tap_fraction median=0.4279885986870768
slip_ratio         median=0.26504220129124356
tracking_ratio     median=0.9269098900992947  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}
facing_error_rms   median=0.5938214216444141  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.7788829977963998 (n=100)
       fwd: cmd 0.25 -> achieved 0.243 m/s | tripod median=0.594022796418536 (n=100)
     lat_l: cmd 0.22 -> achieved 0.224 m/s | tripod median=0.6445913923433024 (n=100)
      diag: cmd 0.25 -> achieved 0.248 m/s | tripod median=0.6567378133504989 (n=100)
      back: cmd 0.22 -> achieved 0.212 m/s | tripod median=0.6953130611304998 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.007 m/s | tripod median=0.78237164084288 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.012 m/s | tripod median=0.7920732882808765 (n=100)
  move_turn: cmd 0.22 -> achieved 0.157 m/s | tripod median=0.6135129913851562 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.020 initial=0.006 final=0.026 max=0.044 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a
       fwd: err rms=0.057 initial=0.027 final=0.063 max=0.118 closed=n/a | settle=0.000 (n=98/100) | progress=-0.001 net=0.009 yaw_rate_ratio=0.345
     lat_l: err rms=0.046 initial=0.064 final=0.030 max=0.094 closed=0.729 | settle=0.000 (n=100/100) | progress=0.002 net=0.006 yaw_rate_ratio=0.426
      diag: err rms=0.045 initial=0.029 final=0.041 max=0.085 closed=n/a | settle=0.000 (n=100/100) | progress=0.023 net=0.001 yaw_rate_ratio=0.374
      back: err rms=0.060 initial=0.041 final=0.089 max=0.120 closed=n/a | settle=0.000 (n=98/100) | progress=-0.000 net=0.016 yaw_rate_ratio=0.562
  turn_l90: err rms=0.752 initial=1.482 final=0.015 max=1.491 closed=0.990 | settle=7.920 (n=100/100) | progress=0.170 net=0.123 yaw_rate_ratio=0.644
  turn_r90: err rms=0.777 initial=1.567 final=0.037 max=1.571 closed=0.977 | settle=8.040 (n=100/100) | progress=0.174 net=0.127 yaw_rate_ratio=0.649
  move_turn: err rms=1.220 initial=1.536 final=0.949 max=1.547 closed=0.387 | settle=n/a (n=0/100) | progress=0.074 net=0.075 yaw_rate_ratio=0.408

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
