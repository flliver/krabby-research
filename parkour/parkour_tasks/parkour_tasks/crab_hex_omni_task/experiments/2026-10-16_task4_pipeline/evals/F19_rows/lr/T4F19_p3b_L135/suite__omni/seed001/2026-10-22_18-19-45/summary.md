```
=== crab-hex gait eval ===
scenario   : suite__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6691675013415026  p25=0.620472265812174  p75=0.7230817625933608
tippy_tap_fraction median=0.4142857142857143
slip_ratio         median=0.204147980100632
tracking_ratio     median=1.0343542880617416  (achieved/commanded vx, walking holds; n=700)
schedule_completion_rate=0.998
terminations={'schedule_complete': 998, 'fall': 2}
facing_error_rms   median=0.042611800332553254  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.6206830692231161 (n=960)
     stand: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.7348593815074012 (n=100)
       fwd: cmd 0.25 -> achieved 0.260 m/s | tripod median=0.6475493954042131 (n=100)
     lat_l: cmd 0.22 -> achieved 0.233 m/s | tripod median=0.6589484137089952 (n=100)
     lat_r: cmd 0.22 -> achieved 0.231 m/s | tripod median=0.7584567053281492 (n=100)
      back: cmd 0.22 -> achieved 0.226 m/s | tripod median=0.6935047830266261 (n=100)
      diag: cmd 0.25 -> achieved 0.261 m/s | tripod median=0.6520760775948446 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.010 m/s | tripod median=0.7478256587839716 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.009 m/s | tripod median=0.7334514416341182 (n=100)
  move_turn: cmd 0.22 -> achieved 0.176 m/s | tripod median=0.5368582672516443 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.174 m/s | tripod median=0.6032204053609769 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.011 initial=0.007 final=0.012 max=0.025 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.002 yaw_rate_ratio=n/a
     stand: err rms=0.022 initial=0.012 final=0.026 max=0.050 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a
       fwd: err rms=0.035 initial=0.011 final=0.029 max=0.084 closed=n/a | settle=0.000 (n=100/100) | progress=0.003 net=0.001 yaw_rate_ratio=1.097
     lat_l: err rms=0.018 initial=0.012 final=0.014 max=0.043 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
     lat_r: err rms=0.025 initial=0.011 final=0.029 max=0.059 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.002 yaw_rate_ratio=n/a
      back: err rms=0.055 initial=0.013 final=0.067 max=0.122 closed=n/a | settle=0.000 (n=98/100) | progress=-0.002 net=0.002 yaw_rate_ratio=0.553
      diag: err rms=0.033 initial=0.011 final=0.028 max=0.079 closed=n/a | settle=0.000 (n=100/100) | progress=-0.000 net=0.002 yaw_rate_ratio=0.366
  turn_l90: err rms=0.792 initial=1.571 final=0.015 max=1.576 closed=0.990 | settle=8.410 (n=100/100) | progress=0.170 net=0.131 yaw_rate_ratio=0.638
  turn_r90: err rms=0.787 initial=1.571 final=0.022 max=1.577 closed=0.986 | settle=8.480 (n=100/100) | progress=0.169 net=0.130 yaw_rate_ratio=0.641
  move_turn: err rms=1.177 initial=1.573 final=0.850 max=1.583 closed=0.459 | settle=n/a (n=0/100) | progress=0.059 net=0.060 yaw_rate_ratio=0.444
  move_turn_r: err rms=1.004 initial=1.569 final=0.390 max=1.575 closed=0.749 | settle=11.940 (n=1/100) | progress=0.097 net=0.098 yaw_rate_ratio=0.476

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.004 tripod=0.735 | closed=n/a progress=n/a net=0.000 | drift=0.031 dir_err=n/a
        fwd: n=100 surv=1.000 (fwd=1.00) | ratio=1.039 achieved=0.260 tripod=0.648 | closed=n/a progress=0.003 net=0.001 | drift=0.259 dir_err=0.009
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=1.054 achieved=0.232 tripod=0.692 | closed=n/a progress=n/a net=0.001 | drift=0.233 dir_err=0.021
       back: n=100 surv=1.000 (back=1.00) | ratio=1.027 achieved=0.226 tripod=0.694 | closed=n/a progress=-0.002 net=0.002 | drift=0.226 dir_err=0.009
       diag: n=100 surv=1.000 (diag=1.00) | ratio=1.045 achieved=0.261 tripod=0.652 | closed=n/a progress=-0.000 net=0.002 | drift=0.263 dir_err=0.048
       turn: n=200 surv=1.000 (turn_l90=1.00 turn_r90=1.00) | ratio=n/a achieved=0.001 tripod=0.741 | closed=0.989 progress=0.169 net=0.131 | drift=0.047 dir_err=n/a
  move_turn: n=200 surv=0.990 (move_turn=0.98 move_turn_r=1.00) | ratio=0.795 achieved=0.175 tripod=0.560 | closed=0.546 progress=0.071 net=0.072 | drift=0.189 dir_err=0.092

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.155 action_fft_freq=1.536 joint_speed_mean=0.368 rod_speed_mean=0.029 contact_switch_hz=1.306 foot_slip=0.081 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.027 settle_six_down_s=12.000 foot_disp_end=0.098 work_per_s=376.050
        fwd: action_jitter=0.111 action_fft_freq=1.009 joint_speed_mean=0.758 rod_speed_mean=0.029 contact_switch_hz=1.674 foot_slip=0.107 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=19.771 slip_per_m=1.301 work_per_m=1431.547 cam_work_per_m=544.556
       side: action_jitter=0.087 action_fft_freq=1.049 joint_speed_mean=0.387 rod_speed_mean=0.029 contact_switch_hz=1.278 foot_slip=0.089 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=16.738 slip_per_m=1.243 work_per_m=1409.889 cam_work_per_m=250.713
       back: action_jitter=0.115 action_fft_freq=1.050 joint_speed_mean=0.712 rod_speed_mean=0.029 contact_switch_hz=1.639 foot_slip=0.108 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=22.028 slip_per_m=1.518 work_per_m=1535.815 cam_work_per_m=509.271
       diag: action_jitter=0.102 action_fft_freq=1.111 joint_speed_mean=0.575 rod_speed_mean=0.029 contact_switch_hz=1.625 foot_slip=0.110 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=18.785 slip_per_m=1.302 work_per_m=1294.031 cam_work_per_m=269.418
       turn: action_jitter=0.116 action_fft_freq=1.050 joint_speed_mean=0.688 rod_speed_mean=0.029 contact_switch_hz=1.333 foot_slip=0.091 wz_active_frac=0.710 clock_restarts=0.000 swings_per_rad=30.798 slip_per_rad=2.263 work_per_rad=2430.543 cam_work_per_rad=750.324
  move_turn: action_jitter=0.102 action_fft_freq=1.023 joint_speed_mean=0.679 rod_speed_mean=0.028 contact_switch_hz=1.660 foot_slip=0.115 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=65.289 slip_per_rad=4.778 work_per_rad=5255.900 cam_work_per_rad=2147.526

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.323 steps_per_s=2.417 gait_hz=0.397 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=10.054 support_feet=3.112 taps_per_m=9.294 swing_apex=0.119 step_rate_hz=2.583 gait_hz=0.385 clock_hz=0.357 cadence_ratio=1.077 hz_speed_slope=-0.264 offset_R=0.086
       side: steps_per_m=10.549 support_feet=3.193 taps_per_m=5.620 swing_apex=0.099 step_rate_hz=2.417 gait_hz=0.394 clock_hz=0.314 cadence_ratio=1.253 hz_speed_slope=1.217 offset_R=0.021
       back: steps_per_m=11.961 support_feet=3.162 taps_per_m=9.882 swing_apex=0.108 step_rate_hz=2.667 gait_hz=0.409 clock_hz=0.314 cadence_ratio=1.301 hz_speed_slope=-0.332 offset_R=0.045
       diag: steps_per_m=11.437 support_feet=3.083 taps_per_m=7.445 swing_apex=0.093 step_rate_hz=3.000 gait_hz=0.452 clock_hz=0.357 cadence_ratio=1.265 hz_speed_slope=0.198 offset_R=0.221
       turn: steps_per_rad=19.400 support_feet=3.245 taps_per_rad=11.141 swing_apex=0.137 step_rate_hz=2.500 gait_hz=0.409 clock_hz=0.351 cadence_ratio=1.173 steps_per_rad_pre=13.390 steps_per_rad_post=5.813 offset_R=0.051
  move_turn: steps_per_rad=36.039 support_feet=3.105 taps_per_rad=31.162 swing_apex=0.105 step_rate_hz=2.667 gait_hz=0.366 clock_hz=0.500 cadence_ratio=0.732 hz_speed_slope=-0.140 offset_R=0.092

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
