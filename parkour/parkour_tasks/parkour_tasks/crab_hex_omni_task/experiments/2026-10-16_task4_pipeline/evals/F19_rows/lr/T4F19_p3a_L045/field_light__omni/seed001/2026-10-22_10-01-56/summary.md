```
=== crab-hex gait eval ===
scenario   : field_light__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_03-27-53_seed19_X07_p3a_A15pB/model_22993.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7558043894371693  p25=0.743872926858453  p75=0.7637267500825193
tippy_tap_fraction median=0.37795705326665074
slip_ratio         median=0.22392940823379515
tracking_ratio     median=0.9404412727603202  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}
facing_error_rms   median=0.6008258171957106  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.000 m/s | tripod median=0.8216358328569549 (n=100)
       fwd: cmd 0.25 -> achieved 0.252 m/s | tripod median=0.7539068613341435 (n=100)
     lat_l: cmd 0.22 -> achieved 0.206 m/s | tripod median=0.7091514525300322 (n=100)
      diag: cmd 0.25 -> achieved 0.239 m/s | tripod median=0.6824811955229848 (n=100)
      back: cmd 0.22 -> achieved 0.229 m/s | tripod median=0.7298141243234003 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.009 m/s | tripod median=0.8134748673074479 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7999405428931428 (n=100)
  move_turn: cmd 0.22 -> achieved 0.169 m/s | tripod median=0.7010329960352372 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.020 initial=0.011 final=0.021 max=0.048 closed=n/a | settle=0.000 (n=100/100) | progress=0.067 net=0.001 yaw_rate_ratio=0.430
       fwd: err rms=0.064 initial=0.021 final=0.111 max=0.126 closed=n/a | settle=0.000 (n=89/100) | progress=-0.007 net=-0.004 yaw_rate_ratio=0.330
     lat_l: err rms=0.064 initial=0.112 final=0.025 max=0.134 closed=0.862 | settle=0.000 (n=100/100) | progress=0.008 net=0.011 yaw_rate_ratio=0.361
      diag: err rms=0.031 initial=0.025 final=0.026 max=0.071 closed=n/a | settle=0.000 (n=100/100) | progress=-0.004 net=0.002 yaw_rate_ratio=0.492
      back: err rms=0.063 initial=0.024 final=0.107 max=0.130 closed=n/a | settle=0.000 (n=90/100) | progress=-0.008 net=0.013 yaw_rate_ratio=0.361
  turn_l90: err rms=0.974 initial=1.679 final=0.132 max=1.694 closed=0.921 | settle=11.640 (n=69/100) | progress=0.135 net=0.128 yaw_rate_ratio=0.583
  turn_r90: err rms=0.717 initial=1.437 final=0.121 max=1.448 closed=0.916 | settle=11.160 (n=85/100) | progress=0.125 net=0.109 yaw_rate_ratio=0.579
  move_turn: err rms=1.055 initial=1.450 final=0.643 max=1.463 closed=0.559 | settle=n/a (n=0/100) | progress=0.102 net=0.102 yaw_rate_ratio=0.422

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
