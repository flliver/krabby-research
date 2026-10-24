```
=== crab-hex gait eval ===
scenario   : field_hard__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_27000.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6351855070925289  p25=0.5974659642151231  p75=0.6700333979113425
tippy_tap_fraction median=0.38718799030095563
slip_ratio         median=0.27273898404140506
tracking_ratio     median=0.8650389735083404  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.84
terminations={'schedule_complete': 84, 'fall': 16}
facing_error_rms   median=0.5995073230315571  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7680196741327219 (n=100)
       fwd: cmd 0.25 -> achieved 0.242 m/s | tripod median=0.6382823262622543 (n=100)
     lat_l: cmd 0.22 -> achieved 0.216 m/s | tripod median=0.5486862825163535 (n=100)
      diag: cmd 0.25 -> achieved 0.226 m/s | tripod median=0.5177390985848632 (n=95)
      back: cmd 0.22 -> achieved 0.207 m/s | tripod median=0.6142456635876303 (n=90)
  turn_l90: cmd 0.00 -> achieved -0.007 m/s | tripod median=0.7417843358731486 (n=86)
  turn_r90: cmd 0.00 -> achieved 0.008 m/s | tripod median=0.7663593122542656 (n=86)
  move_turn: cmd 0.22 -> achieved 0.148 m/s | tripod median=0.5501454135822568 (n=86)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.020 initial=0.006 final=0.018 max=0.044 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a
       fwd: err rms=0.061 initial=0.018 final=0.064 max=0.118 closed=n/a | settle=0.000 (n=95/100) | progress=-0.003 net=0.009 yaw_rate_ratio=0.653
     lat_l: err rms=0.056 initial=0.063 final=0.034 max=0.108 closed=0.704 | settle=0.000 (n=100/100) | progress=0.002 net=0.005 yaw_rate_ratio=0.417
      diag: err rms=0.055 initial=0.034 final=0.061 max=0.101 closed=n/a | settle=0.000 (n=87/100) | progress=-0.005 net=0.003 yaw_rate_ratio=0.518
      back: err rms=0.066 initial=0.057 final=0.076 max=0.144 closed=0.599 | settle=0.000 (n=82/90) | progress=-0.000 net=0.011 yaw_rate_ratio=0.522
  turn_l90: err rms=0.780 initial=1.500 final=0.023 max=1.508 closed=0.985 | settle=8.500 (n=85/87) | progress=0.162 net=0.124 yaw_rate_ratio=0.634
  turn_r90: err rms=0.789 initial=1.568 final=0.036 max=1.570 closed=0.977 | settle=8.360 (n=85/86) | progress=0.171 net=0.127 yaw_rate_ratio=0.637
  move_turn: err rms=1.204 initial=1.539 final=0.904 max=1.544 closed=0.415 | settle=n/a (n=0/86) | progress=0.078 net=0.080 yaw_rate_ratio=0.415

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
