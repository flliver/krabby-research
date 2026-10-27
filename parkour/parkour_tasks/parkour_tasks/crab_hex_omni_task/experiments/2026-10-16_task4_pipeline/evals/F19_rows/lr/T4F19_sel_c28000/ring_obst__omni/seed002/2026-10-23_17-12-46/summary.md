```
=== crab-hex gait eval ===
scenario   : ring_obst__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_28000.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7025145105918005  p25=0.6916736129285103  p75=0.7135223469283702
tippy_tap_fraction median=0.3921604160100851
slip_ratio         median=0.2565905693675899
tracking_ratio     median=0.9351194190916308  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}
facing_error_rms   median=0.5957399668829928  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.7778015556073301 (n=100)
       fwd: cmd 0.25 -> achieved 0.243 m/s | tripod median=0.5761325603888225 (n=100)
     lat_l: cmd 0.22 -> achieved 0.228 m/s | tripod median=0.6680633863279043 (n=100)
      diag: cmd 0.25 -> achieved 0.251 m/s | tripod median=0.6728828070956057 (n=100)
      back: cmd 0.22 -> achieved 0.215 m/s | tripod median=0.6866139166075702 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.009 m/s | tripod median=0.7736895970138596 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.008 m/s | tripod median=0.7872801286202227 (n=100)
  move_turn: cmd 0.22 -> achieved 0.156 m/s | tripod median=0.6091431682971378 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.026 initial=0.008 final=0.030 max=0.053 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.054 initial=0.030 final=0.050 max=0.110 closed=n/a | settle=0.000 (n=99/100) | progress=0.001 net=0.009 yaw_rate_ratio=0.379
     lat_l: err rms=0.039 initial=0.051 final=0.028 max=0.084 closed=0.723 | settle=0.000 (n=100/100) | progress=-0.000 net=0.004 yaw_rate_ratio=0.426
      diag: err rms=0.043 initial=0.029 final=0.044 max=0.085 closed=n/a | settle=0.000 (n=100/100) | progress=-0.006 net=-0.000 yaw_rate_ratio=0.425
      back: err rms=0.054 initial=0.045 final=0.084 max=0.114 closed=n/a | settle=0.000 (n=99/100) | progress=-0.000 net=0.015 yaw_rate_ratio=0.441
  turn_l90: err rms=0.762 initial=1.486 final=0.023 max=1.495 closed=0.984 | settle=8.020 (n=100/100) | progress=0.168 net=0.124 yaw_rate_ratio=0.639
  turn_r90: err rms=0.787 initial=1.561 final=0.037 max=1.566 closed=0.976 | settle=8.260 (n=100/100) | progress=0.173 net=0.128 yaw_rate_ratio=0.645
  move_turn: err rms=1.214 initial=1.542 final=0.919 max=1.547 closed=0.409 | settle=n/a (n=0/100) | progress=0.079 net=0.080 yaw_rate_ratio=0.410

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
