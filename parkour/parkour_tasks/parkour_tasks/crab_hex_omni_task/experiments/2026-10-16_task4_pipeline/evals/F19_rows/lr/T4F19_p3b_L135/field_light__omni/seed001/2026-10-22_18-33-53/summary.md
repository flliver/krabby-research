```
=== crab-hex gait eval ===
scenario   : field_light__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6602754961865447  p25=0.6440480291993627  p75=0.6732826686868846
tippy_tap_fraction median=0.35796104148240826
slip_ratio         median=0.23833661697192593
tracking_ratio     median=0.9550092225341349  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.87
terminations={'schedule_complete': 87, 'fall': 13}
facing_error_rms   median=0.6050483993775386  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.7525192183020064 (n=100)
       fwd: cmd 0.25 -> achieved 0.253 m/s | tripod median=0.5532054324630571 (n=100)
     lat_l: cmd 0.22 -> achieved 0.223 m/s | tripod median=0.6481297042377564 (n=100)
      diag: cmd 0.25 -> achieved 0.249 m/s | tripod median=0.6113984353402966 (n=100)
      back: cmd 0.22 -> achieved 0.220 m/s | tripod median=0.6271839798821923 (n=96)
  turn_l90: cmd 0.00 -> achieved -0.011 m/s | tripod median=0.7663752176825006 (n=93)
  turn_r90: cmd 0.00 -> achieved 0.011 m/s | tripod median=0.75257144673276 (n=93)
  move_turn: cmd 0.22 -> achieved 0.177 m/s | tripod median=0.5154249686811322 (n=93)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.017 initial=0.007 final=0.020 max=0.041 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a
       fwd: err rms=0.053 initial=0.020 final=0.046 max=0.107 closed=n/a | settle=0.000 (n=97/100) | progress=-0.003 net=0.000 yaw_rate_ratio=0.702
     lat_l: err rms=0.033 initial=0.046 final=0.020 max=0.075 closed=0.939 | settle=0.000 (n=100/100) | progress=0.002 net=0.006 yaw_rate_ratio=0.706
      diag: err rms=0.039 initial=0.021 final=0.035 max=0.081 closed=n/a | settle=0.000 (n=97/100) | progress=-0.025 net=0.002 yaw_rate_ratio=0.520
      back: err rms=0.056 initial=0.035 final=0.065 max=0.124 closed=0.738 | settle=0.000 (n=92/96) | progress=-0.004 net=0.011 yaw_rate_ratio=0.660
  turn_l90: err rms=0.789 initial=1.511 final=0.022 max=1.524 closed=0.985 | settle=8.280 (n=93/94) | progress=0.165 net=0.125 yaw_rate_ratio=0.618
  turn_r90: err rms=0.782 initial=1.559 final=0.042 max=1.562 closed=0.973 | settle=8.420 (n=93/93) | progress=0.169 net=0.126 yaw_rate_ratio=0.637
  move_turn: err rms=1.233 initial=1.533 final=1.008 max=1.536 closed=0.347 | settle=n/a (n=0/93) | progress=0.066 net=0.067 yaw_rate_ratio=0.440

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
