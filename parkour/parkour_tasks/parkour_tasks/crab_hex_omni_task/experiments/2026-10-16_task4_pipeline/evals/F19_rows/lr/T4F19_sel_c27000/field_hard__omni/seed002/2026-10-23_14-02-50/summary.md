```
=== crab-hex gait eval ===
scenario   : field_hard__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_27000.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6278565425286602  p25=0.5929797604752759  p75=0.6645577277589443
tippy_tap_fraction median=0.3866323024054983
slip_ratio         median=0.2720728546731276
tracking_ratio     median=0.8634459448819157  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.87
terminations={'schedule_complete': 87, 'fall': 13}
facing_error_rms   median=0.6004576202489791  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7666949871112703 (n=100)
       fwd: cmd 0.25 -> achieved 0.243 m/s | tripod median=0.6367478056160895 (n=100)
     lat_l: cmd 0.22 -> achieved 0.204 m/s | tripod median=0.5426577854887923 (n=100)
      diag: cmd 0.25 -> achieved 0.227 m/s | tripod median=0.5246290859610684 (n=96)
      back: cmd 0.22 -> achieved 0.204 m/s | tripod median=0.6025475518437505 (n=89)
  turn_l90: cmd 0.00 -> achieved -0.008 m/s | tripod median=0.7314925593524745 (n=88)
  turn_r90: cmd 0.00 -> achieved 0.008 m/s | tripod median=0.7442426682419797 (n=88)
  move_turn: cmd 0.22 -> achieved 0.146 m/s | tripod median=0.5471342139223991 (n=88)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.019 initial=0.007 final=0.023 max=0.045 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.002 yaw_rate_ratio=n/a
       fwd: err rms=0.055 initial=0.023 final=0.064 max=0.118 closed=n/a | settle=0.000 (n=97/100) | progress=-0.003 net=0.009 yaw_rate_ratio=0.497
     lat_l: err rms=0.054 initial=0.064 final=0.027 max=0.110 closed=0.827 | settle=0.000 (n=100/100) | progress=0.002 net=0.005 yaw_rate_ratio=0.466
      diag: err rms=0.056 initial=0.027 final=0.059 max=0.111 closed=n/a | settle=0.000 (n=85/98) | progress=-0.008 net=0.004 yaw_rate_ratio=0.498
      back: err rms=0.066 initial=0.053 final=0.070 max=0.132 closed=0.641 | settle=0.000 (n=82/89) | progress=0.002 net=0.010 yaw_rate_ratio=0.477
  turn_l90: err rms=0.773 initial=1.502 final=0.020 max=1.508 closed=0.987 | settle=8.420 (n=88/88) | progress=0.162 net=0.125 yaw_rate_ratio=0.635
  turn_r90: err rms=0.793 initial=1.567 final=0.031 max=1.572 closed=0.980 | settle=8.400 (n=87/88) | progress=0.169 net=0.129 yaw_rate_ratio=0.642
  move_turn: err rms=1.223 initial=1.553 final=0.939 max=1.562 closed=0.398 | settle=n/a (n=0/88) | progress=0.077 net=0.078 yaw_rate_ratio=0.415

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
