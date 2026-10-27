```
=== crab-hex gait eval ===
scenario   : field_turn__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_03-27-53_seed19_X07_p3a_A15pB/model_22993.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7906104482644603  p25=0.7741369201108241  p75=0.8044057042927057
tippy_tap_fraction median=0.4128720238095238
slip_ratio         median=0.24824384127489735
tracking_ratio     median=0.9927450377392428  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}
facing_error_rms   median=0.5603627632605075  (rad, whole episode; n=100)

by hold:
       fwd: cmd 0.25 -> achieved 0.248 m/s | tripod median=0.7759259037918784 (n=100)
     stand: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.8164689598603677 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.006 m/s | tripod median=0.80279423950705 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.7868160442619329 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
       fwd: err rms=0.105 initial=0.008 final=0.117 max=0.160 closed=n/a | settle=8.620 (n=91/100) | progress=-0.001 net=0.010 yaw_rate_ratio=0.302
     stand: err rms=0.087 initial=0.116 final=0.062 max=0.140 closed=0.611 | settle=0.000 (n=100/100) | progress=0.001 net=0.006 yaw_rate_ratio=0.476
  turn_l90: err rms=0.771 initial=1.509 final=0.116 max=1.512 closed=0.923 | settle=10.840 (n=82/100) | progress=0.131 net=0.115 yaw_rate_ratio=0.593
  turn_r90: err rms=0.725 initial=1.455 final=0.111 max=1.464 closed=0.923 | settle=10.810 (n=88/100) | progress=0.130 net=0.111 yaw_rate_ratio=0.581

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
