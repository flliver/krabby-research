```
=== crab-hex gait eval ===
scenario   : field_light__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7034688078135574  p25=0.6883995384249098  p75=0.7156642546792806
tippy_tap_fraction median=0.42198040948593313
slip_ratio         median=0.2617348855249346
tracking_ratio     median=0.9258586816701568  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.99
terminations={'schedule_complete': 99, 'fall': 1}
facing_error_rms   median=0.5932892542461718  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.7808016920704837 (n=100)
       fwd: cmd 0.25 -> achieved 0.243 m/s | tripod median=0.5763036376648374 (n=100)
     lat_l: cmd 0.22 -> achieved 0.226 m/s | tripod median=0.639473372108001 (n=100)
      diag: cmd 0.25 -> achieved 0.245 m/s | tripod median=0.6517206525140793 (n=100)
      back: cmd 0.22 -> achieved 0.211 m/s | tripod median=0.6919199088730463 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.008 m/s | tripod median=0.7872050159876633 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.011 m/s | tripod median=0.7947261952627843 (n=100)
  move_turn: cmd 0.22 -> achieved 0.157 m/s | tripod median=0.6086958938996161 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.022 initial=0.008 final=0.026 max=0.050 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.058 initial=0.026 final=0.072 max=0.121 closed=n/a | settle=0.000 (n=94/100) | progress=-0.002 net=0.008 yaw_rate_ratio=0.490
     lat_l: err rms=0.054 initial=0.073 final=0.033 max=0.113 closed=0.768 | settle=0.000 (n=100/100) | progress=0.006 net=0.005 yaw_rate_ratio=0.467
      diag: err rms=0.047 initial=0.033 final=0.045 max=0.091 closed=n/a | settle=0.000 (n=100/100) | progress=-0.001 net=-0.000 yaw_rate_ratio=0.338
      back: err rms=0.058 initial=0.046 final=0.088 max=0.122 closed=n/a | settle=0.000 (n=97/100) | progress=-0.004 net=0.015 yaw_rate_ratio=0.534
  turn_l90: err rms=0.757 initial=1.483 final=0.024 max=1.493 closed=0.984 | settle=7.870 (n=100/100) | progress=0.170 net=0.123 yaw_rate_ratio=0.646
  turn_r90: err rms=0.776 initial=1.565 final=0.036 max=1.569 closed=0.977 | settle=7.900 (n=99/100) | progress=0.177 net=0.127 yaw_rate_ratio=0.651
  move_turn: err rms=1.202 initial=1.534 final=0.905 max=1.542 closed=0.407 | settle=n/a (n=0/100) | progress=0.077 net=0.078 yaw_rate_ratio=0.412

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
