```
=== crab-hex gait eval ===
scenario   : field_hard__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_30991.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6435428030015364  p25=0.6008721932557786  p75=0.6709190159309206
tippy_tap_fraction median=0.38782777190598416
slip_ratio         median=0.25984076498694847
tracking_ratio     median=0.8882519781485168  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.9
terminations={'fall': 10, 'schedule_complete': 90}
facing_error_rms   median=0.5955083553452206  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.008 m/s | tripod median=0.7815695891634022 (n=100)
       fwd: cmd 0.25 -> achieved 0.244 m/s | tripod median=0.5425158614044137 (n=100)
     lat_l: cmd 0.22 -> achieved 0.216 m/s | tripod median=0.5959404545540621 (n=99)
      diag: cmd 0.25 -> achieved 0.238 m/s | tripod median=0.5751985083617404 (n=95)
      back: cmd 0.22 -> achieved 0.201 m/s | tripod median=0.6099370315086673 (n=92)
  turn_l90: cmd 0.00 -> achieved -0.006 m/s | tripod median=0.740516821048053 (n=92)
  turn_r90: cmd 0.00 -> achieved 0.015 m/s | tripod median=0.7503143621311043 (n=92)
  move_turn: cmd 0.22 -> achieved 0.149 m/s | tripod median=0.5658553992208216 (n=92)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.020 initial=0.007 final=0.026 max=0.047 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.057 initial=0.027 final=0.050 max=0.121 closed=n/a | settle=0.000 (n=95/100) | progress=-0.002 net=0.007 yaw_rate_ratio=0.452
     lat_l: err rms=0.047 initial=0.049 final=0.035 max=0.096 closed=0.855 | settle=0.000 (n=100/100) | progress=-0.001 net=0.003 yaw_rate_ratio=0.476
      diag: err rms=0.053 initial=0.035 final=0.051 max=0.101 closed=n/a | settle=0.000 (n=91/98) | progress=-0.003 net=0.004 yaw_rate_ratio=0.409
      back: err rms=0.064 initial=0.050 final=0.083 max=0.125 closed=0.336 | settle=0.000 (n=88/93) | progress=0.000 net=0.012 yaw_rate_ratio=0.570
  turn_l90: err rms=0.766 initial=1.489 final=0.020 max=1.497 closed=0.987 | settle=8.110 (n=92/92) | progress=0.165 net=0.124 yaw_rate_ratio=0.632
  turn_r90: err rms=0.786 initial=1.565 final=0.053 max=1.571 closed=0.966 | settle=8.480 (n=91/92) | progress=0.168 net=0.125 yaw_rate_ratio=0.634
  move_turn: err rms=1.200 initial=1.517 final=0.922 max=1.517 closed=0.392 | settle=n/a (n=0/92) | progress=0.074 net=0.075 yaw_rate_ratio=0.416

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
