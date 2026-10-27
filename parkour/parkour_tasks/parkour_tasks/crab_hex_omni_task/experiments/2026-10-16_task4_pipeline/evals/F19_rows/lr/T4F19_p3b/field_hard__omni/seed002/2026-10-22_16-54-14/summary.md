```
=== crab-hex gait eval ===
scenario   : field_hard__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6538901237746448  p25=0.617386895514786  p75=0.6767479985709586
tippy_tap_fraction median=0.42474673442131666
slip_ratio         median=0.2716014742332641
tracking_ratio     median=0.8949899524774998  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.9
terminations={'schedule_complete': 90, 'fall': 10}
facing_error_rms   median=0.5974959914124869  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.005 m/s | tripod median=0.7792436693295592 (n=100)
       fwd: cmd 0.25 -> achieved 0.243 m/s | tripod median=0.5772330256415015 (n=100)
     lat_l: cmd 0.22 -> achieved 0.218 m/s | tripod median=0.6153862070360867 (n=100)
      diag: cmd 0.25 -> achieved 0.232 m/s | tripod median=0.5635513449056242 (n=100)
      back: cmd 0.22 -> achieved 0.207 m/s | tripod median=0.6295153544341052 (n=94)
  turn_l90: cmd 0.00 -> achieved -0.008 m/s | tripod median=0.739210556412573 (n=92)
  turn_r90: cmd 0.00 -> achieved 0.011 m/s | tripod median=0.754269333066947 (n=92)
  move_turn: cmd 0.22 -> achieved 0.150 m/s | tripod median=0.5557470610122337 (n=92)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.020 initial=0.006 final=0.021 max=0.045 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a
       fwd: err rms=0.057 initial=0.021 final=0.060 max=0.113 closed=n/a | settle=0.000 (n=93/100) | progress=-0.009 net=0.007 yaw_rate_ratio=0.542
     lat_l: err rms=0.048 initial=0.061 final=0.024 max=0.107 closed=0.873 | settle=0.000 (n=100/100) | progress=0.000 net=0.005 yaw_rate_ratio=0.570
      diag: err rms=0.052 initial=0.024 final=0.049 max=0.103 closed=n/a | settle=0.000 (n=94/100) | progress=-0.003 net=0.003 yaw_rate_ratio=0.471
      back: err rms=0.062 initial=0.049 final=0.071 max=0.123 closed=0.641 | settle=0.000 (n=90/95) | progress=-0.001 net=0.012 yaw_rate_ratio=0.626
  turn_l90: err rms=0.783 initial=1.503 final=0.030 max=1.505 closed=0.980 | settle=8.280 (n=91/94) | progress=0.164 net=0.123 yaw_rate_ratio=0.632
  turn_r90: err rms=0.778 initial=1.560 final=0.036 max=1.562 closed=0.977 | settle=8.290 (n=90/92) | progress=0.172 net=0.127 yaw_rate_ratio=0.645
  move_turn: err rms=1.206 initial=1.537 final=0.949 max=1.544 closed=0.383 | settle=n/a (n=0/92) | progress=0.072 net=0.074 yaw_rate_ratio=0.411

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
