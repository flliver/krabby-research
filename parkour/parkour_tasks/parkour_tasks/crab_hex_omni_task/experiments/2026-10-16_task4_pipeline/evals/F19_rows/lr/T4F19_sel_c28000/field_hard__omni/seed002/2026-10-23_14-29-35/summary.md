```
=== crab-hex gait eval ===
scenario   : field_hard__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_28000.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6496749973404344  p25=0.6239203475792425  p75=0.6762383212962584
tippy_tap_fraction median=0.388887987012987
slip_ratio         median=0.25876228471711193
tracking_ratio     median=0.8862918995556115  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.88
terminations={'schedule_complete': 88, 'fall': 12}
facing_error_rms   median=0.5950281240373845  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.7735731131691437 (n=100)
       fwd: cmd 0.25 -> achieved 0.240 m/s | tripod median=0.5657249802873954 (n=100)
     lat_l: cmd 0.22 -> achieved 0.217 m/s | tripod median=0.6163845770228757 (n=100)
      diag: cmd 0.25 -> achieved 0.235 m/s | tripod median=0.5814252054151935 (n=97)
      back: cmd 0.22 -> achieved 0.209 m/s | tripod median=0.6278875360696323 (n=92)
  turn_l90: cmd 0.00 -> achieved -0.011 m/s | tripod median=0.7425079205434111 (n=91)
  turn_r90: cmd 0.00 -> achieved 0.008 m/s | tripod median=0.7570887096979567 (n=91)
  move_turn: cmd 0.22 -> achieved 0.150 m/s | tripod median=0.5732171923652747 (n=91)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.023 initial=0.007 final=0.028 max=0.049 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.060 initial=0.028 final=0.045 max=0.125 closed=n/a | settle=0.000 (n=98/100) | progress=-0.002 net=0.008 yaw_rate_ratio=0.504
     lat_l: err rms=0.040 initial=0.043 final=0.035 max=0.086 closed=0.635 | settle=0.000 (n=100/100) | progress=0.003 net=0.004 yaw_rate_ratio=0.609
      diag: err rms=0.045 initial=0.034 final=0.042 max=0.094 closed=n/a | settle=0.000 (n=93/100) | progress=-0.005 net=0.002 yaw_rate_ratio=0.440
      back: err rms=0.065 initial=0.041 final=0.085 max=0.130 closed=0.684 | settle=0.000 (n=84/92) | progress=-0.001 net=0.010 yaw_rate_ratio=0.510
  turn_l90: err rms=0.778 initial=1.492 final=0.025 max=1.501 closed=0.984 | settle=8.370 (n=90/91) | progress=0.162 net=0.124 yaw_rate_ratio=0.632
  turn_r90: err rms=0.781 initial=1.561 final=0.041 max=1.565 closed=0.974 | settle=8.320 (n=90/91) | progress=0.170 net=0.127 yaw_rate_ratio=0.637
  move_turn: err rms=1.185 initial=1.531 final=0.914 max=1.537 closed=0.398 | settle=n/a (n=0/91) | progress=0.074 net=0.076 yaw_rate_ratio=0.421

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
