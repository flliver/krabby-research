```
=== crab-hex gait eval ===
scenario   : field_light__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_28000.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6155365280926024  p25=0.5890681913644551  p75=0.6382584936998523
tippy_tap_fraction median=0.3194444444444444
slip_ratio         median=0.24389524938551044
tracking_ratio     median=0.9581320660348287  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.62
terminations={'schedule_complete': 62, 'fall': 38}
facing_error_rms   median=0.5996664980234698  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.7405107307856873 (n=100)
       fwd: cmd 0.25 -> achieved 0.251 m/s | tripod median=0.5374636358999929 (n=100)
     lat_l: cmd 0.22 -> achieved 0.218 m/s | tripod median=0.6210309890019727 (n=96)
      diag: cmd 0.25 -> achieved 0.249 m/s | tripod median=0.5948749662796259 (n=95)
      back: cmd 0.22 -> achieved 0.222 m/s | tripod median=0.5489526733916401 (n=88)
  turn_l90: cmd 0.00 -> achieved -0.014 m/s | tripod median=0.7292724373089402 (n=77)
  turn_r90: cmd 0.00 -> achieved 0.009 m/s | tripod median=0.7295875507124528 (n=77)
  move_turn: cmd 0.22 -> achieved 0.183 m/s | tripod median=0.403756572111131 (n=76)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.019 initial=0.008 final=0.023 max=0.045 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.000 yaw_rate_ratio=n/a
       fwd: err rms=0.057 initial=0.023 final=0.055 max=0.123 closed=n/a | settle=0.000 (n=99/100) | progress=0.000 net=0.002 yaw_rate_ratio=0.524
     lat_l: err rms=0.041 initial=0.049 final=0.026 max=0.090 closed=0.968 | settle=0.000 (n=96/96) | progress=0.009 net=0.006 yaw_rate_ratio=0.781
      diag: err rms=0.036 initial=0.027 final=0.032 max=0.081 closed=n/a | settle=0.000 (n=91/96) | progress=-0.014 net=0.002 yaw_rate_ratio=0.381
      back: err rms=0.058 initial=0.030 final=0.070 max=0.123 closed=0.537 | settle=0.000 (n=85/90) | progress=-0.001 net=0.007 yaw_rate_ratio=0.540
  turn_l90: err rms=0.795 initial=1.513 final=0.022 max=1.522 closed=0.986 | settle=8.420 (n=76/79) | progress=0.163 net=0.125 yaw_rate_ratio=0.608
  turn_r90: err rms=0.795 initial=1.564 final=0.039 max=1.566 closed=0.975 | settle=8.400 (n=75/77) | progress=0.167 net=0.126 yaw_rate_ratio=0.634
  move_turn: err rms=1.242 initial=1.531 final=1.039 max=1.542 closed=0.326 | settle=n/a (n=0/76) | progress=0.062 net=0.064 yaw_rate_ratio=0.464

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
