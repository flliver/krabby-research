```
=== crab-hex gait eval ===
scenario   : field_hard__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_03-27-53_seed19_X07_p3a_A15pB/model_22993.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.704761584606482  p25=0.6742903892837603  p75=0.7364246674245952
tippy_tap_fraction median=0.41125418533757563
slip_ratio         median=0.2580493952950431
tracking_ratio     median=0.9046380834535143  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.88
terminations={'schedule_complete': 88, 'fall': 12}
facing_error_rms   median=0.5979074930044955  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.8334187693053313 (n=100)
       fwd: cmd 0.25 -> achieved 0.248 m/s | tripod median=0.765184894005804 (n=100)
     lat_l: cmd 0.22 -> achieved 0.198 m/s | tripod median=0.687251782262521 (n=100)
      diag: cmd 0.25 -> achieved 0.229 m/s | tripod median=0.6129055579109381 (n=98)
      back: cmd 0.22 -> achieved 0.219 m/s | tripod median=0.6656802529164736 (n=94)
  turn_l90: cmd 0.00 -> achieved -0.006 m/s | tripod median=0.7568704951677578 (n=92)
  turn_r90: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.7484219925597004 (n=89)
  move_turn: cmd 0.22 -> achieved 0.159 m/s | tripod median=0.6619853932608998 (n=89)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.020 initial=0.011 final=0.019 max=0.048 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.061 initial=0.020 final=0.099 max=0.127 closed=n/a | settle=0.000 (n=92/100) | progress=-0.007 net=-0.006 yaw_rate_ratio=0.320
     lat_l: err rms=0.062 initial=0.101 final=0.029 max=0.133 closed=0.838 | settle=0.000 (n=99/100) | progress=0.004 net=0.008 yaw_rate_ratio=0.393
      diag: err rms=0.040 initial=0.029 final=0.030 max=0.086 closed=-0.069 | settle=0.000 (n=88/99) | progress=-0.014 net=0.003 yaw_rate_ratio=0.483
      back: err rms=0.074 initial=0.030 final=0.092 max=0.141 closed=0.543 | settle=0.000 (n=86/94) | progress=-0.001 net=0.012 yaw_rate_ratio=0.435
  turn_l90: err rms=0.964 initial=1.660 final=0.124 max=1.682 closed=0.924 | settle=11.370 (n=68/92) | progress=0.134 net=0.128 yaw_rate_ratio=0.572
  turn_r90: err rms=0.730 initial=1.449 final=0.117 max=1.455 closed=0.916 | settle=10.800 (n=73/89) | progress=0.125 net=0.110 yaw_rate_ratio=0.570
  move_turn: err rms=1.058 initial=1.454 final=0.646 max=1.465 closed=0.554 | settle=n/a (n=0/89) | progress=0.099 net=0.100 yaw_rate_ratio=0.425

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
