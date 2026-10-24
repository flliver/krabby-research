```
=== crab-hex gait eval ===
scenario   : ring_obst__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_28000.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7012297074651432  p25=0.6883110920015051  p75=0.7183504642911969
tippy_tap_fraction median=0.3907489816580726
slip_ratio         median=0.2545157997477666
tracking_ratio     median=0.935560776695513  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}
facing_error_rms   median=0.5914960606921447  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.7791838361614805 (n=100)
       fwd: cmd 0.25 -> achieved 0.242 m/s | tripod median=0.5647312165841815 (n=100)
     lat_l: cmd 0.22 -> achieved 0.227 m/s | tripod median=0.659018834059486 (n=100)
      diag: cmd 0.25 -> achieved 0.250 m/s | tripod median=0.6710153459229147 (n=100)
      back: cmd 0.22 -> achieved 0.215 m/s | tripod median=0.6951142879823502 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.008 m/s | tripod median=0.7739924345915552 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.010 m/s | tripod median=0.790649869248838 (n=100)
  move_turn: cmd 0.22 -> achieved 0.156 m/s | tripod median=0.6149011739625616 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.023 initial=0.008 final=0.030 max=0.049 closed=n/a | settle=0.000 (n=100/100) | progress=0.024 net=-0.001 yaw_rate_ratio=0.166
       fwd: err rms=0.059 initial=0.029 final=0.051 max=0.116 closed=n/a | settle=0.000 (n=99/100) | progress=-0.001 net=0.008 yaw_rate_ratio=0.417
     lat_l: err rms=0.040 initial=0.051 final=0.026 max=0.091 closed=0.631 | settle=0.000 (n=100/100) | progress=0.003 net=0.004 yaw_rate_ratio=0.477
      diag: err rms=0.041 initial=0.026 final=0.048 max=0.082 closed=n/a | settle=0.000 (n=100/100) | progress=-0.004 net=-0.001 yaw_rate_ratio=0.429
      back: err rms=0.060 initial=0.047 final=0.096 max=0.120 closed=n/a | settle=0.000 (n=98/100) | progress=-0.002 net=0.017 yaw_rate_ratio=0.488
  turn_l90: err rms=0.735 initial=1.474 final=0.022 max=1.480 closed=0.985 | settle=7.940 (n=100/100) | progress=0.167 net=0.122 yaw_rate_ratio=0.643
  turn_r90: err rms=0.774 initial=1.560 final=0.030 max=1.562 closed=0.981 | settle=7.930 (n=100/100) | progress=0.177 net=0.127 yaw_rate_ratio=0.652
  move_turn: err rms=1.203 initial=1.542 final=0.903 max=1.551 closed=0.413 | settle=n/a (n=0/100) | progress=0.079 net=0.080 yaw_rate_ratio=0.408

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
