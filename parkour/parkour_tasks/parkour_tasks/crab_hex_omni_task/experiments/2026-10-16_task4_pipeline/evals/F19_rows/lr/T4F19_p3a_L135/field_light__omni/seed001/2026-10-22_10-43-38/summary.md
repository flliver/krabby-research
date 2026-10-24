```
=== crab-hex gait eval ===
scenario   : field_light__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_03-27-53_seed19_X07_p3a_A15pB/model_22993.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6826524439226382  p25=0.6610359888563222  p75=0.6993107208257014
tippy_tap_fraction median=0.3653548463919981
slip_ratio         median=0.22896924235159538
tracking_ratio     median=0.9644561656775075  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.81
terminations={'schedule_complete': 81, 'fall': 19}
facing_error_rms   median=0.5940332216862324  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved -0.000 m/s | tripod median=0.7525986648881925 (n=100)
       fwd: cmd 0.25 -> achieved 0.255 m/s | tripod median=0.6799616348469423 (n=100)
     lat_l: cmd 0.22 -> achieved 0.214 m/s | tripod median=0.6572974668181599 (n=99)
      diag: cmd 0.25 -> achieved 0.248 m/s | tripod median=0.6049519009847701 (n=97)
      back: cmd 0.22 -> achieved 0.230 m/s | tripod median=0.6488790919042967 (n=90)
  turn_l90: cmd 0.00 -> achieved -0.008 m/s | tripod median=0.7782047859743816 (n=83)
  turn_r90: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.7679225301783317 (n=83)
  move_turn: cmd 0.22 -> achieved 0.179 m/s | tripod median=0.60037264725845 (n=83)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.016 initial=0.008 final=0.014 max=0.042 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.060 initial=0.013 final=0.093 max=0.119 closed=n/a | settle=0.000 (n=85/100) | progress=-0.007 net=-0.005 yaw_rate_ratio=0.385
     lat_l: err rms=0.060 initial=0.092 final=0.021 max=0.131 closed=0.918 | settle=0.000 (n=99/99) | progress=0.009 net=0.009 yaw_rate_ratio=0.412
      diag: err rms=0.036 initial=0.019 final=0.032 max=0.080 closed=n/a | settle=0.000 (n=94/97) | progress=0.000 net=0.004 yaw_rate_ratio=0.533
      back: err rms=0.062 initial=0.031 final=0.088 max=0.125 closed=n/a | settle=0.000 (n=80/90) | progress=-0.006 net=0.003 yaw_rate_ratio=0.419
  turn_l90: err rms=0.972 initial=1.656 final=0.156 max=1.675 closed=0.905 | settle=11.580 (n=35/83) | progress=0.127 net=0.125 yaw_rate_ratio=0.550
  turn_r90: err rms=0.711 initial=1.415 final=0.124 max=1.427 closed=0.911 | settle=11.220 (n=65/83) | progress=0.116 net=0.107 yaw_rate_ratio=0.548
  move_turn: err rms=1.060 initial=1.447 final=0.696 max=1.460 closed=0.515 | settle=n/a (n=0/83) | progress=0.092 net=0.092 yaw_rate_ratio=0.427

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
