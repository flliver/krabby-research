```
=== crab-hex gait eval ===
scenario   : field_hard__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_30991.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6517692439434852  p25=0.6148718107134669  p75=0.6733993009751703
tippy_tap_fraction median=0.3854604163844104
slip_ratio         median=0.2682989905865764
tracking_ratio     median=0.8949283037878908  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.86
terminations={'schedule_complete': 86, 'fall': 14}
facing_error_rms   median=0.5928438603375018  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.008 m/s | tripod median=0.7840683975190891 (n=100)
       fwd: cmd 0.25 -> achieved 0.245 m/s | tripod median=0.5423651634399849 (n=100)
     lat_l: cmd 0.22 -> achieved 0.219 m/s | tripod median=0.5945398661471406 (n=99)
      diag: cmd 0.25 -> achieved 0.238 m/s | tripod median=0.5560465184002312 (n=98)
      back: cmd 0.22 -> achieved 0.205 m/s | tripod median=0.643389522368958 (n=92)
  turn_l90: cmd 0.00 -> achieved -0.005 m/s | tripod median=0.7550874868324875 (n=90)
  turn_r90: cmd 0.00 -> achieved 0.014 m/s | tripod median=0.7630082526146883 (n=89)
  move_turn: cmd 0.22 -> achieved 0.148 m/s | tripod median=0.5535397955661225 (n=89)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.018 initial=0.007 final=0.020 max=0.047 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.000 yaw_rate_ratio=n/a
       fwd: err rms=0.057 initial=0.021 final=0.046 max=0.115 closed=n/a | settle=0.000 (n=96/100) | progress=-0.002 net=0.005 yaw_rate_ratio=0.645
     lat_l: err rms=0.045 initial=0.048 final=0.035 max=0.092 closed=0.643 | settle=0.000 (n=100/100) | progress=-0.001 net=0.003 yaw_rate_ratio=0.546
      diag: err rms=0.052 initial=0.035 final=0.057 max=0.098 closed=n/a | settle=0.000 (n=91/98) | progress=-0.011 net=0.002 yaw_rate_ratio=0.457
      back: err rms=0.062 initial=0.056 final=0.081 max=0.120 closed=0.708 | settle=0.000 (n=88/92) | progress=-0.001 net=0.011 yaw_rate_ratio=0.494
  turn_l90: err rms=0.759 initial=1.490 final=0.024 max=1.496 closed=0.984 | settle=8.010 (n=88/90) | progress=0.169 net=0.124 yaw_rate_ratio=0.645
  turn_r90: err rms=0.791 initial=1.565 final=0.043 max=1.570 closed=0.973 | settle=8.340 (n=88/89) | progress=0.170 net=0.126 yaw_rate_ratio=0.638
  move_turn: err rms=1.199 initial=1.530 final=0.913 max=1.533 closed=0.397 | settle=n/a (n=0/89) | progress=0.074 net=0.075 yaw_rate_ratio=0.420

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
