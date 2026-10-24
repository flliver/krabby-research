```
=== crab-hex gait eval ===
scenario   : ring_obst__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-21_23-06-12_seed19_X06_p2a_A15pB/model_19994.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7173111168502182  p25=0.7031018244069989  p75=0.731602431952699
tippy_tap_fraction median=0.41383011685552407
slip_ratio         median=0.24332046147361847
tracking_ratio     median=0.9655933773226021  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.99
terminations={'schedule_complete': 99, 'fall': 1}
facing_error_rms   median=0.5867736815247026  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.8161278271032393 (n=100)
       fwd: cmd 0.25 -> achieved 0.263 m/s | tripod median=0.5927009974319415 (n=100)
     lat_l: cmd 0.22 -> achieved 0.215 m/s | tripod median=0.7168669251485363 (n=100)
      diag: cmd 0.25 -> achieved 0.239 m/s | tripod median=0.729171872199605 (n=100)
      back: cmd 0.22 -> achieved 0.233 m/s | tripod median=0.7011744825460042 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.003 m/s | tripod median=0.7799179069429112 (n=100)
  turn_r90: cmd 0.00 -> achieved -0.001 m/s | tripod median=0.780729812671751 (n=100)
  move_turn: cmd 0.22 -> achieved 0.175 m/s | tripod median=0.5776413449732767 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.021 initial=0.007 final=0.026 max=0.048 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a
       fwd: err rms=0.050 initial=0.026 final=0.040 max=0.093 closed=n/a | settle=0.000 (n=100/100) | progress=0.000 net=0.001 yaw_rate_ratio=0.323
     lat_l: err rms=0.036 initial=0.041 final=0.029 max=0.082 closed=n/a | settle=0.000 (n=100/100) | progress=-0.002 net=0.004 yaw_rate_ratio=0.596
      diag: err rms=0.044 initial=0.029 final=0.034 max=0.083 closed=n/a | settle=0.000 (n=100/100) | progress=0.004 net=0.003 yaw_rate_ratio=0.291
      back: err rms=0.039 initial=0.035 final=0.031 max=0.080 closed=n/a | settle=0.000 (n=100/100) | progress=0.006 net=0.007 yaw_rate_ratio=0.327
  turn_l90: err rms=0.858 initial=1.583 final=0.089 max=1.602 closed=0.944 | settle=9.820 (n=97/100) | progress=0.149 net=0.126 yaw_rate_ratio=0.584
  turn_r90: err rms=0.752 initial=1.482 final=0.106 max=1.492 closed=0.927 | settle=10.440 (n=91/100) | progress=0.137 net=0.114 yaw_rate_ratio=0.589
  move_turn: err rms=1.079 initial=1.465 final=0.750 max=1.470 closed=0.487 | settle=n/a (n=0/100) | progress=0.088 net=0.089 yaw_rate_ratio=0.395

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
