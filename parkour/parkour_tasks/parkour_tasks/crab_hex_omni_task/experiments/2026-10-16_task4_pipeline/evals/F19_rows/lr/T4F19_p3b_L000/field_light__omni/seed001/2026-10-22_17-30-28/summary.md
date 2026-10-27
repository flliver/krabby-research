```
=== crab-hex gait eval ===
scenario   : field_light__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7030536277793205  p25=0.6947786130569708  p75=0.7141017322279394
tippy_tap_fraction median=0.42098033974815874
slip_ratio         median=0.26223940061140116
tracking_ratio     median=0.9285814864592832  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}
facing_error_rms   median=0.589882400014345  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.005 m/s | tripod median=0.7865070213820128 (n=100)
       fwd: cmd 0.25 -> achieved 0.244 m/s | tripod median=0.5808116097223996 (n=100)
     lat_l: cmd 0.22 -> achieved 0.227 m/s | tripod median=0.6477410213129045 (n=100)
      diag: cmd 0.25 -> achieved 0.245 m/s | tripod median=0.6584663168438903 (n=100)
      back: cmd 0.22 -> achieved 0.212 m/s | tripod median=0.6980922964967735 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.008 m/s | tripod median=0.777511099703146 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.012 m/s | tripod median=0.7924330771475716 (n=100)
  move_turn: cmd 0.22 -> achieved 0.156 m/s | tripod median=0.6232504027979506 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.022 initial=0.008 final=0.023 max=0.049 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.000 yaw_rate_ratio=n/a
       fwd: err rms=0.057 initial=0.023 final=0.062 max=0.116 closed=n/a | settle=0.000 (n=98/100) | progress=-0.001 net=0.009 yaw_rate_ratio=0.437
     lat_l: err rms=0.048 initial=0.062 final=0.024 max=0.097 closed=0.841 | settle=0.000 (n=100/100) | progress=0.003 net=0.005 yaw_rate_ratio=0.591
      diag: err rms=0.044 initial=0.024 final=0.033 max=0.086 closed=n/a | settle=0.000 (n=99/100) | progress=-0.002 net=0.001 yaw_rate_ratio=0.366
      back: err rms=0.059 initial=0.032 final=0.086 max=0.119 closed=0.485 | settle=0.000 (n=96/100) | progress=-0.005 net=0.015 yaw_rate_ratio=0.569
  turn_l90: err rms=0.749 initial=1.484 final=0.021 max=1.488 closed=0.985 | settle=7.890 (n=100/100) | progress=0.170 net=0.123 yaw_rate_ratio=0.647
  turn_r90: err rms=0.763 initial=1.560 final=0.035 max=1.564 closed=0.977 | settle=7.800 (n=99/100) | progress=0.179 net=0.127 yaw_rate_ratio=0.655
  move_turn: err rms=1.197 initial=1.538 final=0.913 max=1.549 closed=0.404 | settle=n/a (n=0/100) | progress=0.078 net=0.079 yaw_rate_ratio=0.415

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
