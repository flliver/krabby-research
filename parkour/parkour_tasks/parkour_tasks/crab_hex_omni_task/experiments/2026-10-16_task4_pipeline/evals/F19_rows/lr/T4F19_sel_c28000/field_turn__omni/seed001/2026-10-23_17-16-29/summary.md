```
=== crab-hex gait eval ===
scenario   : field_turn__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_28000.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7177890338017564  p25=0.6976574784818296  p75=0.7355290859388971
tippy_tap_fraction median=0.39598429951690817
slip_ratio         median=0.2659929222009057
tracking_ratio     median=0.9622780915436205  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}
facing_error_rms   median=0.5916665083518593  (rad, whole episode; n=100)

by hold:
       fwd: cmd 0.25 -> achieved 0.241 m/s | tripod median=0.6019117387641705 (n=100)
     stand: cmd 0.00 -> achieved -0.000 m/s | tripod median=0.739945086055636 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.005 m/s | tripod median=0.7561861598515563 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.008 m/s | tripod median=0.7751501439552992 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
       fwd: err rms=0.057 initial=0.044 final=0.053 max=0.125 closed=n/a | settle=0.000 (n=94/100) | progress=-0.003 net=0.010 yaw_rate_ratio=0.776
     stand: err rms=0.038 initial=0.054 final=0.026 max=0.083 closed=0.897 | settle=0.000 (n=100/100) | progress=0.041 net=0.006 yaw_rate_ratio=0.507
  turn_l90: err rms=0.808 initial=1.568 final=0.021 max=1.571 closed=0.987 | settle=8.540 (n=100/100) | progress=0.167 net=0.130 yaw_rate_ratio=0.653
  turn_r90: err rms=0.777 initial=1.561 final=0.037 max=1.565 closed=0.976 | settle=8.030 (n=100/100) | progress=0.176 net=0.127 yaw_rate_ratio=0.646

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
