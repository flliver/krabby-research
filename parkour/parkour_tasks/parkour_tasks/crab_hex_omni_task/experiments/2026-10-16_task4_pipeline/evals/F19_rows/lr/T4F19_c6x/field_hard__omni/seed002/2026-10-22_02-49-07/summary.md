```
=== crab-hex gait eval ===
scenario   : field_hard__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-10-21_20-59-46_seed19_X05_c6x_A15pB/model_14995.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5984473635074122  p25=0.5540420471243106  p75=0.6289267712744453
tippy_tap_fraction median=0.4010050251256282
slip_ratio         median=0.26611984163800917
tracking_ratio     median=0.8605546064692191  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.49
terminations={'fall': 51, 'schedule_complete': 49}
facing_error_rms   median=0.6220321187000457  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.7579630476426931 (n=100)
       fwd: cmd 0.25 -> achieved 0.240 m/s | tripod median=0.5410070068628514 (n=100)
     lat_l: cmd 0.22 -> achieved 0.217 m/s | tripod median=0.7196051139876597 (n=98)
      diag: cmd 0.25 -> achieved 0.225 m/s | tripod median=0.4772906549780039 (n=90)
      back: cmd 0.22 -> achieved 0.198 m/s | tripod median=0.44288723776273137 (n=77)
  turn_l90: cmd 0.00 -> achieved 0.012 m/s | tripod median=0.6843216224941071 (n=60)
  turn_r90: cmd 0.00 -> achieved -0.002 m/s | tripod median=0.7057654691293228 (n=60)
  move_turn: cmd 0.22 -> achieved 0.124 m/s | tripod median=0.43755088275597226 (n=60)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.019 initial=0.011 final=0.015 max=0.040 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.050 initial=0.014 final=0.045 max=0.105 closed=n/a | settle=0.000 (n=99/100) | progress=-0.003 net=0.004 yaw_rate_ratio=0.526
     lat_l: err rms=0.039 initial=0.042 final=0.021 max=0.089 closed=n/a | settle=0.000 (n=99/99) | progress=-0.000 net=0.004 yaw_rate_ratio=0.524
      diag: err rms=0.055 initial=0.021 final=0.049 max=0.105 closed=n/a | settle=0.000 (n=85/94) | progress=-0.005 net=0.001 yaw_rate_ratio=0.472
      back: err rms=0.064 initial=0.047 final=0.084 max=0.140 closed=0.267 | settle=0.000 (n=66/79) | progress=-0.001 net=0.013 yaw_rate_ratio=0.628
  turn_l90: err rms=1.231 initial=1.638 final=0.851 max=1.656 closed=0.467 | settle=n/a (n=0/63) | progress=0.064 net=0.063 yaw_rate_ratio=0.417
  turn_r90: err rms=0.307 initial=0.731 final=0.032 max=0.746 closed=0.957 | settle=4.920 (n=60/60) | progress=0.103 net=0.059 yaw_rate_ratio=0.445
  move_turn: err rms=1.208 initial=1.547 final=0.967 max=1.552 closed=0.371 | settle=n/a (n=0/60) | progress=0.075 net=0.076 yaw_rate_ratio=0.417

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
