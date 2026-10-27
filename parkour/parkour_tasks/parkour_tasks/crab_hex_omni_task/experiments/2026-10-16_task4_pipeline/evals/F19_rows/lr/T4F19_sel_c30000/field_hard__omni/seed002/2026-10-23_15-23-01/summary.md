```
=== crab-hex gait eval ===
scenario   : field_hard__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_30000.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6311558316815236  p25=0.5987842043966444  p75=0.6641967636240821
tippy_tap_fraction median=0.38609645986054
slip_ratio         median=0.26090791232996474
tracking_ratio     median=0.8614803094167557  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.79
terminations={'fall': 21, 'schedule_complete': 79}
facing_error_rms   median=0.580935075167692  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved -0.001 m/s | tripod median=0.8013258715598924 (n=100)
       fwd: cmd 0.25 -> achieved 0.236 m/s | tripod median=0.6256694084781387 (n=100)
     lat_l: cmd 0.22 -> achieved 0.210 m/s | tripod median=0.5478634827446872 (n=100)
      diag: cmd 0.25 -> achieved 0.219 m/s | tripod median=0.44232603912094637 (n=95)
      back: cmd 0.22 -> achieved 0.212 m/s | tripod median=0.6888755022095954 (n=87)
  turn_l90: cmd 0.00 -> achieved -0.008 m/s | tripod median=0.7259639834455613 (n=86)
  turn_r90: cmd 0.00 -> achieved 0.005 m/s | tripod median=0.7461272569211227 (n=86)
  move_turn: cmd 0.22 -> achieved 0.143 m/s | tripod median=0.4763097650586063 (n=86)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.032 initial=0.016 final=0.039 max=0.054 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.003 yaw_rate_ratio=n/a
       fwd: err rms=0.066 initial=0.039 final=0.038 max=0.140 closed=n/a | settle=0.000 (n=97/100) | progress=-0.003 net=0.001 yaw_rate_ratio=0.606
     lat_l: err rms=0.042 initial=0.039 final=0.026 max=0.089 closed=-0.025 | settle=0.000 (n=98/100) | progress=0.002 net=0.002 yaw_rate_ratio=0.392
      diag: err rms=0.044 initial=0.026 final=0.048 max=0.100 closed=0.021 | settle=0.000 (n=84/99) | progress=-0.011 net=0.002 yaw_rate_ratio=0.463
      back: err rms=0.093 initial=0.044 final=0.114 max=0.157 closed=0.677 | settle=1.470 (n=76/87) | progress=-0.001 net=0.008 yaw_rate_ratio=0.396
  turn_l90: err rms=0.726 initial=1.459 final=0.030 max=1.473 closed=0.979 | settle=7.870 (n=84/86) | progress=0.166 net=0.123 yaw_rate_ratio=0.619
  turn_r90: err rms=0.794 initial=1.564 final=0.079 max=1.567 closed=0.949 | settle=8.480 (n=85/86) | progress=0.167 net=0.124 yaw_rate_ratio=0.640
  move_turn: err rms=1.162 initial=1.492 final=0.928 max=1.502 closed=0.382 | settle=n/a (n=0/86) | progress=0.072 net=0.073 yaw_rate_ratio=0.428

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
