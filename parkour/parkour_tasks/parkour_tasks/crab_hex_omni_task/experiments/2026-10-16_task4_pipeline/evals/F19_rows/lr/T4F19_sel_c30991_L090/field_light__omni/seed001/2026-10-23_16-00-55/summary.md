```
=== crab-hex gait eval ===
scenario   : field_light__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_30991.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6781372477151684  p25=0.6653312312576029  p75=0.6891884914148567
tippy_tap_fraction median=0.32178023701638203
slip_ratio         median=0.22562590022033074
tracking_ratio     median=0.9638797403697228  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.93
terminations={'schedule_complete': 93, 'fall': 7}
facing_error_rms   median=0.601336293456227  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.008 m/s | tripod median=0.7715753708333686 (n=100)
       fwd: cmd 0.25 -> achieved 0.252 m/s | tripod median=0.5311863387299377 (n=100)
     lat_l: cmd 0.22 -> achieved 0.230 m/s | tripod median=0.6592316581992701 (n=99)
      diag: cmd 0.25 -> achieved 0.259 m/s | tripod median=0.6385774100227447 (n=99)
      back: cmd 0.22 -> achieved 0.213 m/s | tripod median=0.6566550089959005 (n=95)
  turn_l90: cmd 0.00 -> achieved -0.007 m/s | tripod median=0.7700414064552847 (n=94)
  turn_r90: cmd 0.00 -> achieved 0.017 m/s | tripod median=0.7726308353424768 (n=94)
  move_turn: cmd 0.22 -> achieved 0.172 m/s | tripod median=0.5348946109126906 (n=94)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.018 initial=0.006 final=0.024 max=0.045 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a
       fwd: err rms=0.056 initial=0.024 final=0.036 max=0.113 closed=n/a | settle=0.000 (n=98/100) | progress=0.001 net=0.002 yaw_rate_ratio=0.523
     lat_l: err rms=0.033 initial=0.036 final=0.031 max=0.073 closed=0.973 | settle=0.000 (n=99/99) | progress=0.009 net=0.005 yaw_rate_ratio=0.611
      diag: err rms=0.047 initial=0.029 final=0.051 max=0.087 closed=n/a | settle=0.000 (n=98/99) | progress=-0.002 net=-0.000 yaw_rate_ratio=0.416
      back: err rms=0.052 initial=0.050 final=0.069 max=0.114 closed=n/a | settle=0.000 (n=90/95) | progress=-0.004 net=0.014 yaw_rate_ratio=0.548
  turn_l90: err rms=0.768 initial=1.500 final=0.021 max=1.512 closed=0.986 | settle=7.940 (n=94/94) | progress=0.171 net=0.125 yaw_rate_ratio=0.644
  turn_r90: err rms=0.790 initial=1.568 final=0.043 max=1.571 closed=0.972 | settle=8.320 (n=92/94) | progress=0.171 net=0.127 yaw_rate_ratio=0.643
  move_turn: err rms=1.237 initial=1.528 final=0.981 max=1.538 closed=0.366 | settle=n/a (n=0/94) | progress=0.069 net=0.071 yaw_rate_ratio=0.423

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
