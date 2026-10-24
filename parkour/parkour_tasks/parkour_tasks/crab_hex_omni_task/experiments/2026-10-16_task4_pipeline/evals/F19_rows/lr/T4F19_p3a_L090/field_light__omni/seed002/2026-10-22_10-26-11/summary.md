```
=== crab-hex gait eval ===
scenario   : field_light__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_03-27-53_seed19_X07_p3a_A15pB/model_22993.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7263099656989552  p25=0.712922571726887  p75=0.7380418621231086
tippy_tap_fraction median=0.3674775251189847
slip_ratio         median=0.2236356016838446
tracking_ratio     median=0.9553280682424798  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.95
terminations={'schedule_complete': 95, 'fall': 5}
facing_error_rms   median=0.6008628626484042  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved -0.000 m/s | tripod median=0.7847138751645589 (n=100)
       fwd: cmd 0.25 -> achieved 0.253 m/s | tripod median=0.7243815549342574 (n=100)
     lat_l: cmd 0.22 -> achieved 0.208 m/s | tripod median=0.6878487831630122 (n=100)
      diag: cmd 0.25 -> achieved 0.245 m/s | tripod median=0.6402185124843842 (n=100)
      back: cmd 0.22 -> achieved 0.229 m/s | tripod median=0.7027691291281672 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.009 m/s | tripod median=0.801440933104059 (n=97)
  turn_r90: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.7851360329334859 (n=97)
  move_turn: cmd 0.22 -> achieved 0.176 m/s | tripod median=0.6687243884999383 (n=97)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.019 initial=0.008 final=0.024 max=0.047 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.064 initial=0.023 final=0.104 max=0.128 closed=n/a | settle=0.000 (n=92/100) | progress=-0.005 net=-0.004 yaw_rate_ratio=0.323
     lat_l: err rms=0.063 initial=0.106 final=0.021 max=0.134 closed=0.880 | settle=0.000 (n=100/100) | progress=0.004 net=0.010 yaw_rate_ratio=0.384
      diag: err rms=0.029 initial=0.020 final=0.024 max=0.070 closed=n/a | settle=0.000 (n=99/100) | progress=-0.005 net=0.002 yaw_rate_ratio=0.407
      back: err rms=0.061 initial=0.024 final=0.109 max=0.128 closed=0.671 | settle=0.000 (n=93/100) | progress=-0.010 net=0.010 yaw_rate_ratio=0.451
  turn_l90: err rms=0.972 initial=1.679 final=0.148 max=1.697 closed=0.912 | settle=11.580 (n=52/98) | progress=0.131 net=0.127 yaw_rate_ratio=0.571
  turn_r90: err rms=0.718 initial=1.425 final=0.117 max=1.436 closed=0.917 | settle=11.260 (n=83/97) | progress=0.119 net=0.109 yaw_rate_ratio=0.557
  move_turn: err rms=1.050 initial=1.455 final=0.651 max=1.463 closed=0.547 | settle=n/a (n=0/97) | progress=0.101 net=0.101 yaw_rate_ratio=0.428

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
