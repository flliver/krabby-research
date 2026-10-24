```
=== crab-hex gait eval ===
scenario   : field_light__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6016381423917954  p25=0.5571770209314263  p75=0.6196810969113673
tippy_tap_fraction median=0.37270207340059874
slip_ratio         median=0.2525256743733672
tracking_ratio     median=0.94685920698831  (achieved/commanded vx, walking holds; n=98)
schedule_completion_rate=0.37
terminations={'fall': 63, 'schedule_complete': 37}
facing_error_rms   median=0.5838209506749514  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.7266707249849083 (n=100)
       fwd: cmd 0.25 -> achieved 0.249 m/s | tripod median=0.48536823571675547 (n=98)
     lat_l: cmd 0.22 -> achieved 0.218 m/s | tripod median=0.6334957616190353 (n=88)
      diag: cmd 0.25 -> achieved 0.242 m/s | tripod median=0.5603982886767542 (n=84)
      back: cmd 0.22 -> achieved 0.221 m/s | tripod median=0.5722829252928144 (n=73)
  turn_l90: cmd 0.00 -> achieved -0.010 m/s | tripod median=0.7282392639685625 (n=59)
  turn_r90: cmd 0.00 -> achieved 0.011 m/s | tripod median=0.7269460989207485 (n=59)
  move_turn: cmd 0.22 -> achieved 0.186 m/s | tripod median=0.3817680184009222 (n=59)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.017 initial=0.008 final=0.021 max=0.043 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.002 yaw_rate_ratio=n/a
       fwd: err rms=0.053 initial=0.021 final=0.049 max=0.126 closed=n/a | settle=0.000 (n=97/99) | progress=0.002 net=-0.000 yaw_rate_ratio=0.580
     lat_l: err rms=0.035 initial=0.046 final=0.023 max=0.073 closed=0.323 | settle=0.000 (n=89/90) | progress=-0.002 net=0.004 yaw_rate_ratio=0.593
      diag: err rms=0.041 initial=0.022 final=0.044 max=0.085 closed=n/a | settle=0.000 (n=77/84) | progress=-0.024 net=0.001 yaw_rate_ratio=0.479
      back: err rms=0.058 initial=0.043 final=0.066 max=0.130 closed=0.539 | settle=0.000 (n=69/77) | progress=-0.002 net=0.012 yaw_rate_ratio=0.569
  turn_l90: err rms=0.780 initial=1.508 final=0.021 max=1.515 closed=0.987 | settle=8.260 (n=58/59) | progress=0.166 net=0.125 yaw_rate_ratio=0.623
  turn_r90: err rms=0.796 initial=1.562 final=0.048 max=1.564 closed=0.968 | settle=8.610 (n=58/59) | progress=0.166 net=0.126 yaw_rate_ratio=0.628
  move_turn: err rms=1.267 initial=1.522 final=1.087 max=1.528 closed=0.281 | settle=n/a (n=0/59) | progress=0.058 net=0.060 yaw_rate_ratio=0.437

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
