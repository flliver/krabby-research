```
=== crab-hex gait eval ===
scenario   : field_turn__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_03-27-53_seed19_X07_p3a_A15pB/model_22993.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.789530081584968  p25=0.7640191365973908  p75=0.8022167112042107
tippy_tap_fraction median=0.42749597632521674
slip_ratio         median=0.24897483110190066
tracking_ratio     median=0.9947512696872056  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}
facing_error_rms   median=0.562670826605838  (rad, whole episode; n=100)

by hold:
       fwd: cmd 0.25 -> achieved 0.249 m/s | tripod median=0.7667239714271019 (n=100)
     stand: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.8169699739652161 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.005 m/s | tripod median=0.8011762760289869 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.7858327876221802 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
       fwd: err rms=0.103 initial=0.009 final=0.112 max=0.159 closed=n/a | settle=8.620 (n=90/100) | progress=-0.001 net=0.006 yaw_rate_ratio=0.306
     stand: err rms=0.085 initial=0.112 final=0.060 max=0.141 closed=0.597 | settle=0.000 (n=100/100) | progress=0.002 net=0.007 yaw_rate_ratio=0.429
  turn_l90: err rms=0.784 initial=1.511 final=0.120 max=1.514 closed=0.920 | settle=10.810 (n=88/100) | progress=0.133 net=0.115 yaw_rate_ratio=0.588
  turn_r90: err rms=0.721 initial=1.450 final=0.115 max=1.457 closed=0.921 | settle=10.760 (n=92/100) | progress=0.131 net=0.112 yaw_rate_ratio=0.583

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
