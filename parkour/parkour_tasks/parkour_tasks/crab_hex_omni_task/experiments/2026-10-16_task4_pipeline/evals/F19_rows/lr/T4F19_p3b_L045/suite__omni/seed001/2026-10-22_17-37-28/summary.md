```
=== crab-hex gait eval ===
scenario   : suite__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7098247450864161  p25=0.6717208820870381  p75=0.766861206194463
tippy_tap_fraction median=0.4266666666666667
slip_ratio         median=0.2169723420396726
tracking_ratio     median=1.0160073213892464  (achieved/commanded vx, walking holds; n=700)
schedule_completion_rate=1.0
terminations={'schedule_complete': 1000}
facing_error_rms   median=0.04610895303680902  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.7447206540682576 (n=980)
     stand: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.7775350165621563 (n=100)
       fwd: cmd 0.25 -> achieved 0.255 m/s | tripod median=0.6654786402489414 (n=100)
     lat_l: cmd 0.22 -> achieved 0.235 m/s | tripod median=0.6709783431972445 (n=100)
     lat_r: cmd 0.22 -> achieved 0.233 m/s | tripod median=0.762434011870955 (n=100)
      back: cmd 0.22 -> achieved 0.221 m/s | tripod median=0.7051012283536635 (n=100)
      diag: cmd 0.25 -> achieved 0.256 m/s | tripod median=0.6879557144783042 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.006 m/s | tripod median=0.791944328185963 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.007 m/s | tripod median=0.784825254109419 (n=100)
  move_turn: cmd 0.22 -> achieved 0.165 m/s | tripod median=0.6240157341138963 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.170 m/s | tripod median=0.6896632060325872 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.012 initial=0.007 final=0.012 max=0.027 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.002 yaw_rate_ratio=n/a
     stand: err rms=0.023 initial=0.011 final=0.028 max=0.056 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.041 initial=0.012 final=0.040 max=0.094 closed=n/a | settle=0.000 (n=100/100) | progress=-0.001 net=0.001 yaw_rate_ratio=0.355
     lat_l: err rms=0.016 initial=0.014 final=0.013 max=0.041 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
     lat_r: err rms=0.030 initial=0.013 final=0.027 max=0.063 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
      back: err rms=0.048 initial=0.013 final=0.067 max=0.109 closed=n/a | settle=0.000 (n=100/100) | progress=-0.001 net=-0.000 yaw_rate_ratio=0.444
      diag: err rms=0.035 initial=0.011 final=0.033 max=0.081 closed=n/a | settle=0.000 (n=99/100) | progress=-0.001 net=0.001 yaw_rate_ratio=0.296
  turn_l90: err rms=0.782 initial=1.569 final=0.016 max=1.571 closed=0.990 | settle=8.200 (n=100/100) | progress=0.174 net=0.132 yaw_rate_ratio=0.654
  turn_r90: err rms=0.798 initial=1.571 final=0.021 max=1.576 closed=0.987 | settle=8.400 (n=100/100) | progress=0.170 net=0.131 yaw_rate_ratio=0.643
  move_turn: err rms=1.110 initial=1.570 final=0.664 max=1.573 closed=0.577 | settle=n/a (n=0/100) | progress=0.074 net=0.075 yaw_rate_ratio=0.421
  move_turn_r: err rms=1.011 initial=1.574 final=0.373 max=1.578 closed=0.764 | settle=11.700 (n=1/100) | progress=0.101 net=0.101 yaw_rate_ratio=0.458

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.004 tripod=0.778 | closed=n/a progress=n/a net=0.001 | drift=0.031 dir_err=n/a
        fwd: n=100 surv=1.000 (fwd=1.00) | ratio=1.021 achieved=0.255 tripod=0.665 | closed=n/a progress=-0.001 net=0.001 | drift=0.254 dir_err=0.009
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=1.066 achieved=0.235 tripod=0.724 | closed=n/a progress=n/a net=0.001 | drift=0.235 dir_err=0.020
       back: n=100 surv=1.000 (back=1.00) | ratio=1.006 achieved=0.221 tripod=0.705 | closed=n/a progress=-0.001 net=-0.000 | drift=0.222 dir_err=0.012
       diag: n=100 surv=1.000 (diag=1.00) | ratio=1.025 achieved=0.256 tripod=0.688 | closed=n/a progress=-0.001 net=0.001 | drift=0.258 dir_err=0.057
       turn: n=200 surv=1.000 (turn_l90=1.00 turn_r90=1.00) | ratio=n/a achieved=0.002 tripod=0.787 | closed=0.988 progress=0.173 net=0.131 | drift=0.046 dir_err=n/a
  move_turn: n=200 surv=1.000 (move_turn=1.00 move_turn_r=1.00) | ratio=0.765 achieved=0.168 tripod=0.657 | closed=0.658 progress=0.085 net=0.086 | drift=0.182 dir_err=0.135

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.186 action_fft_freq=1.801 joint_speed_mean=0.365 rod_speed_mean=0.029 contact_switch_hz=1.472 foot_slip=0.075 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.017 settle_six_down_s=12.000 foot_disp_end=0.102 work_per_s=317.531
        fwd: action_jitter=0.124 action_fft_freq=1.086 joint_speed_mean=0.757 rod_speed_mean=0.029 contact_switch_hz=1.806 foot_slip=0.100 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=21.524 slip_per_m=1.176 work_per_m=1205.382 cam_work_per_m=444.013
       side: action_jitter=0.097 action_fft_freq=1.124 joint_speed_mean=0.388 rod_speed_mean=0.029 contact_switch_hz=1.333 foot_slip=0.078 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=17.089 slip_per_m=1.042 work_per_m=1124.759 cam_work_per_m=206.120
       back: action_jitter=0.130 action_fft_freq=1.163 joint_speed_mean=0.707 rod_speed_mean=0.029 contact_switch_hz=1.736 foot_slip=0.098 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=23.577 slip_per_m=1.352 work_per_m=1308.401 cam_work_per_m=423.007
       diag: action_jitter=0.112 action_fft_freq=1.188 joint_speed_mean=0.578 rod_speed_mean=0.029 contact_switch_hz=1.674 foot_slip=0.092 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=19.619 slip_per_m=1.086 work_per_m=1098.199 cam_work_per_m=229.471
       turn: action_jitter=0.138 action_fft_freq=1.182 joint_speed_mean=0.683 rod_speed_mean=0.029 contact_switch_hz=1.417 foot_slip=0.077 wz_active_frac=0.697 clock_restarts=0.000 swings_per_rad=32.390 slip_per_rad=1.858 work_per_rad=2011.054 cam_work_per_rad=643.252
  move_turn: action_jitter=0.114 action_fft_freq=1.071 joint_speed_mean=0.683 rod_speed_mean=0.028 contact_switch_hz=1.833 foot_slip=0.106 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=63.845 slip_per_rad=3.678 work_per_rad=3734.922 cam_work_per_rad=1336.510

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.216 steps_per_s=2.500 gait_hz=0.415 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=10.394 support_feet=2.988 taps_per_m=10.770 swing_apex=0.118 step_rate_hz=2.667 gait_hz=0.391 clock_hz=0.357 cadence_ratio=1.094 hz_speed_slope=-0.013 offset_R=0.118
       side: steps_per_m=10.757 support_feet=3.109 taps_per_m=6.403 swing_apex=0.104 step_rate_hz=2.500 gait_hz=0.412 clock_hz=0.314 cadence_ratio=1.311 hz_speed_slope=2.401 offset_R=0.074
       back: steps_per_m=12.310 support_feet=3.042 taps_per_m=11.282 swing_apex=0.111 step_rate_hz=2.667 gait_hz=0.421 clock_hz=0.314 cadence_ratio=1.321 hz_speed_slope=0.122 offset_R=0.179
       diag: steps_per_m=11.580 support_feet=2.989 taps_per_m=8.059 swing_apex=0.092 step_rate_hz=2.917 gait_hz=0.464 clock_hz=0.357 cadence_ratio=1.299 hz_speed_slope=-1.091 offset_R=0.101
       turn: steps_per_rad=19.165 support_feet=3.160 taps_per_rad=12.759 swing_apex=0.139 step_rate_hz=2.500 gait_hz=0.433 clock_hz=0.346 cadence_ratio=1.238 steps_per_rad_pre=13.417 steps_per_rad_post=5.716 offset_R=0.089
  move_turn: steps_per_rad=30.761 support_feet=3.039 taps_per_rad=32.355 swing_apex=0.110 step_rate_hz=2.667 gait_hz=0.366 clock_hz=0.500 cadence_ratio=0.732 hz_speed_slope=0.049 offset_R=0.009

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
