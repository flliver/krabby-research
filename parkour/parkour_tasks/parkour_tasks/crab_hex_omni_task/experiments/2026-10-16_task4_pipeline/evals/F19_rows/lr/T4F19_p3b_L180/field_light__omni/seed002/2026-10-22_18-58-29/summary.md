```
=== crab-hex gait eval ===
scenario   : field_light__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.603855643801338  p25=0.5634465609432335  p75=0.6297723856899817
tippy_tap_fraction median=0.37778568041725935
slip_ratio         median=0.25123627939787396
tracking_ratio     median=0.9500567952210843  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.48
terminations={'fall': 52, 'schedule_complete': 48}
facing_error_rms   median=0.5850421698137815  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.7286219717211833 (n=100)
       fwd: cmd 0.25 -> achieved 0.248 m/s | tripod median=0.47785742721553925 (n=100)
     lat_l: cmd 0.22 -> achieved 0.216 m/s | tripod median=0.6326492995325089 (n=90)
      diag: cmd 0.25 -> achieved 0.244 m/s | tripod median=0.5721839318524259 (n=89)
      back: cmd 0.22 -> achieved 0.221 m/s | tripod median=0.5750038840543367 (n=77)
  turn_l90: cmd 0.00 -> achieved -0.012 m/s | tripod median=0.7343169031161547 (n=64)
  turn_r90: cmd 0.00 -> achieved 0.010 m/s | tripod median=0.7240211692204531 (n=64)
  move_turn: cmd 0.22 -> achieved 0.186 m/s | tripod median=0.42126607760398654 (n=64)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.015 initial=0.009 final=0.011 max=0.038 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.051 initial=0.012 final=0.064 max=0.127 closed=n/a | settle=0.000 (n=93/100) | progress=-0.001 net=0.000 yaw_rate_ratio=0.644
     lat_l: err rms=0.046 initial=0.060 final=0.026 max=0.105 closed=0.828 | settle=0.000 (n=95/96) | progress=0.003 net=0.006 yaw_rate_ratio=0.532
      diag: err rms=0.041 initial=0.026 final=0.043 max=0.090 closed=n/a | settle=0.000 (n=87/90) | progress=-0.008 net=0.001 yaw_rate_ratio=0.446
      back: err rms=0.055 initial=0.037 final=0.062 max=0.123 closed=0.562 | settle=0.000 (n=75/81) | progress=-0.004 net=0.007 yaw_rate_ratio=0.591
  turn_l90: err rms=0.796 initial=1.515 final=0.031 max=1.523 closed=0.980 | settle=8.380 (n=63/66) | progress=0.162 net=0.125 yaw_rate_ratio=0.611
  turn_r90: err rms=0.784 initial=1.546 final=0.047 max=1.548 closed=0.970 | settle=8.500 (n=61/64) | progress=0.165 net=0.125 yaw_rate_ratio=0.625
  move_turn: err rms=1.245 initial=1.524 final=1.053 max=1.535 closed=0.307 | settle=n/a (n=0/64) | progress=0.061 net=0.063 yaw_rate_ratio=0.456

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
