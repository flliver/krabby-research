```
=== crab-hex gait eval ===
scenario   : field_hard__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-21_23-06-12_seed19_X06_p2a_A15pB/model_19994.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6612587581371143  p25=0.6157049966312298  p75=0.6821987386783215
tippy_tap_fraction median=0.409871979805265
slip_ratio         median=0.24566558510895506
tracking_ratio     median=0.9296052710400555  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.73
terminations={'fall': 27, 'schedule_complete': 73}
facing_error_rms   median=0.5901182140403851  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.8227129148537501 (n=100)
       fwd: cmd 0.25 -> achieved 0.255 m/s | tripod median=0.5908962132428228 (n=100)
     lat_l: cmd 0.22 -> achieved 0.207 m/s | tripod median=0.6657802966310038 (n=99)
      diag: cmd 0.25 -> achieved 0.231 m/s | tripod median=0.6384790681516834 (n=97)
      back: cmd 0.22 -> achieved 0.228 m/s | tripod median=0.6197630699994863 (n=92)
  turn_l90: cmd 0.00 -> achieved -0.001 m/s | tripod median=0.7194673833584745 (n=89)
  turn_r90: cmd 0.00 -> achieved -0.002 m/s | tripod median=0.7357693400835381 (n=88)
  move_turn: cmd 0.22 -> achieved 0.160 m/s | tripod median=0.5357662462731416 (n=80)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.024 initial=0.006 final=0.027 max=0.053 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.050 initial=0.027 final=0.047 max=0.095 closed=n/a | settle=0.000 (n=99/100) | progress=-0.001 net=0.004 yaw_rate_ratio=0.341
     lat_l: err rms=0.044 initial=0.048 final=0.028 max=0.094 closed=0.377 | settle=0.000 (n=99/100) | progress=0.004 net=0.003 yaw_rate_ratio=0.632
      diag: err rms=0.045 initial=0.029 final=0.043 max=0.095 closed=n/a | settle=0.000 (n=94/99) | progress=-0.014 net=0.003 yaw_rate_ratio=0.443
      back: err rms=0.050 initial=0.040 final=0.037 max=0.104 closed=0.565 | settle=0.000 (n=88/93) | progress=0.002 net=0.005 yaw_rate_ratio=0.518
  turn_l90: err rms=0.867 initial=1.570 final=0.086 max=1.578 closed=0.944 | settle=10.000 (n=81/90) | progress=0.143 net=0.124 yaw_rate_ratio=0.582
  turn_r90: err rms=0.767 initial=1.485 final=0.106 max=1.489 closed=0.928 | settle=10.680 (n=75/88) | progress=0.133 net=0.115 yaw_rate_ratio=0.573
  move_turn: err rms=1.115 initial=1.465 final=0.838 max=1.472 closed=0.437 | settle=n/a (n=0/81) | progress=0.081 net=0.081 yaw_rate_ratio=0.398

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
