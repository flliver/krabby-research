```
=== crab-hex gait eval ===
scenario   : field_turn__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7208420011931344  p25=0.7013590132128098  p75=0.7341263698739917
tippy_tap_fraction median=0.4366147230288387
slip_ratio         median=0.276144758738266
tracking_ratio     median=0.9663832434548576  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}
facing_error_rms   median=0.5948851521572136  (rad, whole episode; n=100)

by hold:
       fwd: cmd 0.25 -> achieved 0.242 m/s | tripod median=0.593838881187418 (n=100)
     stand: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7492553160168751 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.005 m/s | tripod median=0.7657049316842404 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.010 m/s | tripod median=0.7739746241807995 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
       fwd: err rms=0.060 initial=0.043 final=0.058 max=0.132 closed=n/a | settle=0.000 (n=92/100) | progress=-0.004 net=0.009 yaw_rate_ratio=0.604
     stand: err rms=0.043 initial=0.058 final=0.032 max=0.092 closed=0.746 | settle=0.000 (n=100/100) | progress=0.025 net=0.005 yaw_rate_ratio=0.565
  turn_l90: err rms=0.813 initial=1.580 final=0.018 max=1.587 closed=0.989 | settle=8.480 (n=100/100) | progress=0.170 net=0.131 yaw_rate_ratio=0.653
  turn_r90: err rms=0.783 initial=1.566 final=0.038 max=1.568 closed=0.976 | settle=8.220 (n=99/100) | progress=0.173 net=0.127 yaw_rate_ratio=0.645

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
