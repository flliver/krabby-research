```
=== crab-hex gait eval ===
scenario   : field_hard__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-21_23-06-12_seed19_X06_p2a_A15pB/model_19994.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6546064341723342  p25=0.62806804620813  p75=0.6834920335100922
tippy_tap_fraction median=0.414275679472632
slip_ratio         median=0.2546093272297447
tracking_ratio     median=0.9159676135322368  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.76
terminations={'schedule_complete': 76, 'fall': 24}
facing_error_rms   median=0.5853898460833291  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.8204837296714182 (n=100)
       fwd: cmd 0.25 -> achieved 0.257 m/s | tripod median=0.5775783020557574 (n=100)
     lat_l: cmd 0.22 -> achieved 0.207 m/s | tripod median=0.66514956797651 (n=100)
      diag: cmd 0.25 -> achieved 0.232 m/s | tripod median=0.6253579167901655 (n=99)
      back: cmd 0.22 -> achieved 0.225 m/s | tripod median=0.6430845935836199 (n=87)
  turn_l90: cmd 0.00 -> achieved -0.003 m/s | tripod median=0.727526755399625 (n=86)
  turn_r90: cmd 0.00 -> achieved -0.001 m/s | tripod median=0.7460360672575546 (n=85)
  move_turn: cmd 0.22 -> achieved 0.158 m/s | tripod median=0.5134866411719612 (n=83)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.023 initial=0.006 final=0.023 max=0.051 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.050 initial=0.024 final=0.046 max=0.097 closed=n/a | settle=0.000 (n=96/100) | progress=-0.001 net=0.001 yaw_rate_ratio=0.433
     lat_l: err rms=0.044 initial=0.048 final=0.032 max=0.087 closed=0.802 | settle=0.000 (n=100/100) | progress=0.003 net=0.005 yaw_rate_ratio=0.534
      diag: err rms=0.049 initial=0.032 final=0.043 max=0.101 closed=n/a | settle=0.000 (n=92/100) | progress=-0.002 net=0.002 yaw_rate_ratio=0.406
      back: err rms=0.048 initial=0.038 final=0.050 max=0.107 closed=0.455 | settle=0.000 (n=83/88) | progress=-0.001 net=0.005 yaw_rate_ratio=0.445
  turn_l90: err rms=0.856 initial=1.565 final=0.090 max=1.571 closed=0.943 | settle=9.980 (n=79/86) | progress=0.146 net=0.123 yaw_rate_ratio=0.583
  turn_r90: err rms=0.770 initial=1.483 final=0.107 max=1.490 closed=0.928 | settle=10.430 (n=78/85) | progress=0.136 net=0.115 yaw_rate_ratio=0.577
  move_turn: err rms=1.105 initial=1.466 final=0.800 max=1.470 closed=0.457 | settle=n/a (n=0/83) | progress=0.085 net=0.086 yaw_rate_ratio=0.405

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
