```
=== crab-hex gait eval ===
scenario   : ring_obst__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7039165314908109  p25=0.6931586836889646  p75=0.7119987194328922
tippy_tap_fraction median=0.4254981884057971
slip_ratio         median=0.2610795695851659
tracking_ratio     median=0.9273733311651093  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}
facing_error_rms   median=0.5931810905979173  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.005 m/s | tripod median=0.7834135868000078 (n=100)
       fwd: cmd 0.25 -> achieved 0.243 m/s | tripod median=0.5797858459096644 (n=100)
     lat_l: cmd 0.22 -> achieved 0.225 m/s | tripod median=0.6385488467817166 (n=100)
      diag: cmd 0.25 -> achieved 0.247 m/s | tripod median=0.6619336523205497 (n=100)
      back: cmd 0.22 -> achieved 0.212 m/s | tripod median=0.6957369936995043 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.007 m/s | tripod median=0.7792804399964 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.010 m/s | tripod median=0.7900749899469138 (n=100)
  move_turn: cmd 0.22 -> achieved 0.155 m/s | tripod median=0.6158891391176304 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.017 initial=0.007 final=0.018 max=0.042 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.057 initial=0.019 final=0.073 max=0.113 closed=n/a | settle=0.000 (n=94/100) | progress=-0.004 net=0.006 yaw_rate_ratio=0.498
     lat_l: err rms=0.049 initial=0.074 final=0.025 max=0.104 closed=0.869 | settle=0.000 (n=100/100) | progress=0.002 net=0.006 yaw_rate_ratio=0.463
      diag: err rms=0.044 initial=0.026 final=0.039 max=0.093 closed=n/a | settle=0.000 (n=98/100) | progress=-0.004 net=0.003 yaw_rate_ratio=0.407
      back: err rms=0.060 initial=0.039 final=0.092 max=0.123 closed=0.584 | settle=0.000 (n=96/100) | progress=-0.002 net=0.015 yaw_rate_ratio=0.477
  turn_l90: err rms=0.744 initial=1.478 final=0.023 max=1.486 closed=0.985 | settle=7.820 (n=100/100) | progress=0.170 net=0.122 yaw_rate_ratio=0.649
  turn_r90: err rms=0.781 initial=1.560 final=0.035 max=1.562 closed=0.978 | settle=8.090 (n=100/100) | progress=0.175 net=0.127 yaw_rate_ratio=0.652
  move_turn: err rms=1.220 initial=1.538 final=0.960 max=1.547 closed=0.382 | settle=n/a (n=0/100) | progress=0.073 net=0.073 yaw_rate_ratio=0.405

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
