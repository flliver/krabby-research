```
=== crab-hex gait eval ===
scenario   : field_light__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6988051342311307  p25=0.6891332779232514  p75=0.7068206123268743
tippy_tap_fraction median=0.38531766818828106
slip_ratio         median=0.24866685990488321
tracking_ratio     median=0.9416729470450569  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.97
terminations={'schedule_complete': 97, 'fall': 3}
facing_error_rms   median=0.6054171189211137  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.7843019950646879 (n=100)
       fwd: cmd 0.25 -> achieved 0.248 m/s | tripod median=0.5738850768638266 (n=100)
     lat_l: cmd 0.22 -> achieved 0.228 m/s | tripod median=0.6598933866411534 (n=100)
      diag: cmd 0.25 -> achieved 0.248 m/s | tripod median=0.6559341273163257 (n=100)
      back: cmd 0.22 -> achieved 0.215 m/s | tripod median=0.6913185835852454 (n=99)
  turn_l90: cmd 0.00 -> achieved -0.009 m/s | tripod median=0.7882529507213906 (n=99)
  turn_r90: cmd 0.00 -> achieved 0.011 m/s | tripod median=0.7878460106226938 (n=99)
  move_turn: cmd 0.22 -> achieved 0.161 m/s | tripod median=0.6032539455826306 (n=99)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.019 initial=0.008 final=0.024 max=0.045 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.055 initial=0.024 final=0.042 max=0.113 closed=n/a | settle=0.000 (n=98/100) | progress=-0.002 net=0.005 yaw_rate_ratio=0.344
     lat_l: err rms=0.034 initial=0.045 final=0.028 max=0.083 closed=0.743 | settle=0.000 (n=100/100) | progress=0.003 net=0.004 yaw_rate_ratio=0.526
      diag: err rms=0.047 initial=0.028 final=0.053 max=0.091 closed=n/a | settle=0.000 (n=98/100) | progress=-0.002 net=-0.001 yaw_rate_ratio=0.456
      back: err rms=0.055 initial=0.051 final=0.075 max=0.119 closed=0.743 | settle=0.000 (n=97/99) | progress=-0.002 net=0.016 yaw_rate_ratio=0.585
  turn_l90: err rms=0.761 initial=1.498 final=0.017 max=1.502 closed=0.989 | settle=7.980 (n=99/99) | progress=0.170 net=0.124 yaw_rate_ratio=0.644
  turn_r90: err rms=0.792 initial=1.570 final=0.037 max=1.575 closed=0.976 | settle=8.220 (n=99/99) | progress=0.173 net=0.128 yaw_rate_ratio=0.647
  move_turn: err rms=1.229 initial=1.539 final=0.947 max=1.547 closed=0.386 | settle=n/a (n=0/99) | progress=0.073 net=0.074 yaw_rate_ratio=0.422

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
