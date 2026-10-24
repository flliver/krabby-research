```
=== crab-hex gait eval ===
scenario   : field_turn__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7277245004265542  p25=0.7074108034240238  p75=0.7430011048465183
tippy_tap_fraction median=0.4350844783926273
slip_ratio         median=0.27287912484995763
tracking_ratio     median=0.9624591264840706  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}
facing_error_rms   median=0.597296339232773  (rad, whole episode; n=100)

by hold:
       fwd: cmd 0.25 -> achieved 0.241 m/s | tripod median=0.6110758106482771 (n=100)
     stand: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.7425409753030978 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.005 m/s | tripod median=0.7732664405974221 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.011 m/s | tripod median=0.7728153538575515 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
       fwd: err rms=0.062 initial=0.050 final=0.059 max=0.131 closed=n/a | settle=0.000 (n=96/100) | progress=-0.001 net=0.010 yaw_rate_ratio=0.512
     stand: err rms=0.043 initial=0.059 final=0.027 max=0.087 closed=0.819 | settle=0.000 (n=100/100) | progress=0.012 net=0.005 yaw_rate_ratio=0.429
  turn_l90: err rms=0.806 initial=1.579 final=0.018 max=1.583 closed=0.989 | settle=8.380 (n=99/100) | progress=0.170 net=0.131 yaw_rate_ratio=0.657
  turn_r90: err rms=0.789 initial=1.570 final=0.040 max=1.573 closed=0.974 | settle=8.200 (n=100/100) | progress=0.173 net=0.127 yaw_rate_ratio=0.642

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
