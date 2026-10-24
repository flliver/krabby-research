```
=== crab-hex gait eval ===
scenario   : field_light__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6798613680028526  p25=0.670867133684908  p75=0.6901045666412731
tippy_tap_fraction median=0.36347891566265056
slip_ratio         median=0.23630635879169348
tracking_ratio     median=0.9549687815809109  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.97
terminations={'schedule_complete': 97, 'fall': 3}
facing_error_rms   median=0.60139649181301  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.005 m/s | tripod median=0.7724131902417419 (n=100)
       fwd: cmd 0.25 -> achieved 0.247 m/s | tripod median=0.562899097211361 (n=100)
     lat_l: cmd 0.22 -> achieved 0.228 m/s | tripod median=0.6650576361020246 (n=99)
      diag: cmd 0.25 -> achieved 0.249 m/s | tripod median=0.6316397201301692 (n=99)
      back: cmd 0.22 -> achieved 0.217 m/s | tripod median=0.665644921352133 (n=99)
  turn_l90: cmd 0.00 -> achieved -0.010 m/s | tripod median=0.7720145116570908 (n=98)
  turn_r90: cmd 0.00 -> achieved 0.012 m/s | tripod median=0.7773376807477447 (n=98)
  move_turn: cmd 0.22 -> achieved 0.170 m/s | tripod median=0.549211438563667 (n=98)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.017 initial=0.007 final=0.022 max=0.040 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a
       fwd: err rms=0.060 initial=0.022 final=0.039 max=0.118 closed=n/a | settle=0.000 (n=98/100) | progress=-0.001 net=0.003 yaw_rate_ratio=0.464
     lat_l: err rms=0.033 initial=0.039 final=0.025 max=0.073 closed=0.527 | settle=0.000 (n=99/99) | progress=-0.003 net=0.005 yaw_rate_ratio=0.774
      diag: err rms=0.043 initial=0.025 final=0.049 max=0.086 closed=n/a | settle=0.000 (n=97/99) | progress=-0.001 net=-0.000 yaw_rate_ratio=0.472
      back: err rms=0.055 initial=0.047 final=0.084 max=0.123 closed=0.714 | settle=0.000 (n=95/99) | progress=-0.003 net=0.014 yaw_rate_ratio=0.689
  turn_l90: err rms=0.766 initial=1.489 final=0.031 max=1.500 closed=0.980 | settle=7.970 (n=98/98) | progress=0.168 net=0.123 yaw_rate_ratio=0.637
  turn_r90: err rms=0.777 initial=1.556 final=0.039 max=1.560 closed=0.974 | settle=8.170 (n=98/98) | progress=0.172 net=0.127 yaw_rate_ratio=0.641
  move_turn: err rms=1.236 initial=1.535 final=0.978 max=1.547 closed=0.359 | settle=n/a (n=0/98) | progress=0.068 net=0.069 yaw_rate_ratio=0.428

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
