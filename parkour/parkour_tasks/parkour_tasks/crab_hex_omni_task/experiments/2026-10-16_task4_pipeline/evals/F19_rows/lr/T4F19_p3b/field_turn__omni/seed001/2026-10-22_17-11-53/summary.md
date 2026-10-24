```
=== crab-hex gait eval ===
scenario   : field_turn__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7229320392520535  p25=0.7044316145537399  p75=0.7407359058039742
tippy_tap_fraction median=0.4348484848484848
slip_ratio         median=0.27563945086058506
tracking_ratio     median=0.9631962427257736  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}
facing_error_rms   median=0.5961340863529325  (rad, whole episode; n=100)

by hold:
       fwd: cmd 0.25 -> achieved 0.241 m/s | tripod median=0.6023677442324245 (n=100)
     stand: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.7453588597145313 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.006 m/s | tripod median=0.7724816879240636 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.010 m/s | tripod median=0.7681730840469145 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
       fwd: err rms=0.066 initial=0.045 final=0.065 max=0.133 closed=n/a | settle=0.000 (n=85/100) | progress=-0.005 net=0.011 yaw_rate_ratio=0.654
     stand: err rms=0.046 initial=0.064 final=0.027 max=0.091 closed=0.833 | settle=0.000 (n=99/100) | progress=0.027 net=0.005 yaw_rate_ratio=0.569
  turn_l90: err rms=0.811 initial=1.586 final=0.018 max=1.588 closed=0.989 | settle=8.380 (n=100/100) | progress=0.172 net=0.132 yaw_rate_ratio=0.652
  turn_r90: err rms=0.781 initial=1.566 final=0.041 max=1.575 closed=0.974 | settle=8.200 (n=99/100) | progress=0.173 net=0.127 yaw_rate_ratio=0.646

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
