```
=== crab-hex gait eval ===
scenario   : field_hard__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-21_23-06-12_seed19_X06_p2a_A15pB/model_19994.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6669901516051263  p25=0.625728437552879  p75=0.6965101823708539
tippy_tap_fraction median=0.40696398605911965
slip_ratio         median=0.2506242433837834
tracking_ratio     median=0.9188068707698516  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.82
terminations={'fall': 18, 'schedule_complete': 82}
facing_error_rms   median=0.5822966445160145  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.8155492254391505 (n=100)
       fwd: cmd 0.25 -> achieved 0.260 m/s | tripod median=0.5787584832059077 (n=100)
     lat_l: cmd 0.22 -> achieved 0.209 m/s | tripod median=0.6663630580171788 (n=100)
      diag: cmd 0.25 -> achieved 0.232 m/s | tripod median=0.6409039719529468 (n=98)
      back: cmd 0.22 -> achieved 0.226 m/s | tripod median=0.6587765369565101 (n=90)
  turn_l90: cmd 0.00 -> achieved -0.003 m/s | tripod median=0.7444252193756895 (n=87)
  turn_r90: cmd 0.00 -> achieved -0.001 m/s | tripod median=0.7574208353702443 (n=87)
  move_turn: cmd 0.22 -> achieved 0.164 m/s | tripod median=0.5569434246680662 (n=87)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.022 initial=0.008 final=0.020 max=0.050 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.000 yaw_rate_ratio=n/a
       fwd: err rms=0.049 initial=0.019 final=0.051 max=0.095 closed=n/a | settle=0.000 (n=99/100) | progress=0.002 net=0.005 yaw_rate_ratio=0.275
     lat_l: err rms=0.045 initial=0.051 final=0.033 max=0.091 closed=0.591 | settle=0.000 (n=100/100) | progress=0.001 net=0.004 yaw_rate_ratio=0.389
      diag: err rms=0.051 initial=0.034 final=0.046 max=0.099 closed=-0.409 | settle=0.000 (n=89/99) | progress=-0.009 net=0.000 yaw_rate_ratio=0.470
      back: err rms=0.050 initial=0.044 final=0.041 max=0.108 closed=0.502 | settle=0.000 (n=87/92) | progress=-0.000 net=0.005 yaw_rate_ratio=0.527
  turn_l90: err rms=0.853 initial=1.574 final=0.090 max=1.591 closed=0.942 | settle=9.790 (n=78/87) | progress=0.145 net=0.123 yaw_rate_ratio=0.577
  turn_r90: err rms=0.753 initial=1.479 final=0.108 max=1.487 closed=0.925 | settle=10.300 (n=79/87) | progress=0.136 net=0.115 yaw_rate_ratio=0.584
  move_turn: err rms=1.093 initial=1.462 final=0.778 max=1.470 closed=0.472 | settle=n/a (n=0/87) | progress=0.087 net=0.087 yaw_rate_ratio=0.394

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
