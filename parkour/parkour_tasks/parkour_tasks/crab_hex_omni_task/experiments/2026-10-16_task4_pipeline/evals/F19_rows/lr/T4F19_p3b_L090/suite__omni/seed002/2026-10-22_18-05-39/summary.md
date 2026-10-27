```
=== crab-hex gait eval ===
scenario   : suite__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6916376306672294  p25=0.6547483672176717  p75=0.7505325441014619
tippy_tap_fraction median=0.4142857142857143
slip_ratio         median=0.21554639349992855
tracking_ratio     median=1.0271698956331505  (achieved/commanded vx, walking holds; n=700)
schedule_completion_rate=1.0
terminations={'schedule_complete': 1000}
facing_error_rms   median=0.04243784785796366  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.6901793632039741 (n=974)
     stand: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.7521877751645447 (n=100)
       fwd: cmd 0.25 -> achieved 0.257 m/s | tripod median=0.6728982441974911 (n=100)
     lat_l: cmd 0.22 -> achieved 0.236 m/s | tripod median=0.6696077450057534 (n=100)
     lat_r: cmd 0.22 -> achieved 0.233 m/s | tripod median=0.7688243202834169 (n=100)
      back: cmd 0.22 -> achieved 0.224 m/s | tripod median=0.6940302388479764 (n=100)
      diag: cmd 0.25 -> achieved 0.258 m/s | tripod median=0.670153674880015 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.008 m/s | tripod median=0.7697531022557359 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.008 m/s | tripod median=0.7675928965860005 (n=100)
  move_turn: cmd 0.22 -> achieved 0.170 m/s | tripod median=0.6008916903885073 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.173 m/s | tripod median=0.6508052275634908 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.011 initial=0.007 final=0.012 max=0.026 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.002 yaw_rate_ratio=n/a
     stand: err rms=0.022 initial=0.013 final=0.028 max=0.051 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.034 initial=0.011 final=0.033 max=0.084 closed=n/a | settle=0.000 (n=99/100) | progress=-0.003 net=0.001 yaw_rate_ratio=0.468
     lat_l: err rms=0.018 initial=0.013 final=0.012 max=0.045 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
     lat_r: err rms=0.025 initial=0.011 final=0.027 max=0.057 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.002 yaw_rate_ratio=n/a
      back: err rms=0.048 initial=0.013 final=0.056 max=0.108 closed=n/a | settle=0.000 (n=100/100) | progress=0.001 net=0.001 yaw_rate_ratio=0.574
      diag: err rms=0.030 initial=0.011 final=0.027 max=0.076 closed=n/a | settle=0.000 (n=100/100) | progress=-0.002 net=0.001 yaw_rate_ratio=0.366
  turn_l90: err rms=0.789 initial=1.572 final=0.018 max=1.576 closed=0.989 | settle=8.320 (n=100/100) | progress=0.172 net=0.132 yaw_rate_ratio=0.643
  turn_r90: err rms=0.790 initial=1.570 final=0.020 max=1.574 closed=0.987 | settle=8.420 (n=100/100) | progress=0.170 net=0.130 yaw_rate_ratio=0.643
  move_turn: err rms=1.132 initial=1.571 final=0.733 max=1.577 closed=0.530 | settle=n/a (n=0/100) | progress=0.069 net=0.070 yaw_rate_ratio=0.442
  move_turn_r: err rms=1.009 initial=1.569 final=0.380 max=1.575 closed=0.759 | settle=n/a (n=0/100) | progress=0.099 net=0.100 yaw_rate_ratio=0.471

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.004 tripod=0.752 | closed=n/a progress=n/a net=0.001 | drift=0.030 dir_err=n/a
        fwd: n=100 surv=1.000 (fwd=1.00) | ratio=1.030 achieved=0.257 tripod=0.673 | closed=n/a progress=-0.003 net=0.001 | drift=0.256 dir_err=0.009
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=1.065 achieved=0.234 tripod=0.718 | closed=n/a progress=n/a net=0.001 | drift=0.234 dir_err=0.019
       back: n=100 surv=1.000 (back=1.00) | ratio=1.019 achieved=0.224 tripod=0.694 | closed=n/a progress=0.001 net=0.001 | drift=0.224 dir_err=0.011
       diag: n=100 surv=1.000 (diag=1.00) | ratio=1.032 achieved=0.258 tripod=0.670 | closed=n/a progress=-0.002 net=0.001 | drift=0.260 dir_err=0.049
       turn: n=200 surv=1.000 (turn_l90=1.00 turn_r90=1.00) | ratio=n/a achieved=0.001 tripod=0.768 | closed=0.988 progress=0.171 net=0.131 | drift=0.045 dir_err=n/a
  move_turn: n=200 surv=1.000 (move_turn=1.00 move_turn_r=1.00) | ratio=0.780 achieved=0.172 tripod=0.627 | closed=0.628 progress=0.082 net=0.082 | drift=0.185 dir_err=0.107

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.167 action_fft_freq=1.642 joint_speed_mean=0.366 rod_speed_mean=0.029 contact_switch_hz=1.319 foot_slip=0.078 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.022 settle_six_down_s=12.000 foot_disp_end=0.097 work_per_s=349.930
        fwd: action_jitter=0.116 action_fft_freq=1.033 joint_speed_mean=0.758 rod_speed_mean=0.029 contact_switch_hz=1.694 foot_slip=0.105 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=19.765 slip_per_m=1.250 work_per_m=1309.909 cam_work_per_m=488.982
       side: action_jitter=0.089 action_fft_freq=1.072 joint_speed_mean=0.386 rod_speed_mean=0.029 contact_switch_hz=1.208 foot_slip=0.081 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=15.647 slip_per_m=1.103 work_per_m=1255.347 cam_work_per_m=224.768
       back: action_jitter=0.121 action_fft_freq=1.093 joint_speed_mean=0.707 rod_speed_mean=0.029 contact_switch_hz=1.674 foot_slip=0.102 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=22.587 slip_per_m=1.424 work_per_m=1426.468 cam_work_per_m=478.940
       diag: action_jitter=0.107 action_fft_freq=1.145 joint_speed_mean=0.574 rod_speed_mean=0.029 contact_switch_hz=1.639 foot_slip=0.099 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=18.989 slip_per_m=1.175 work_per_m=1187.405 cam_work_per_m=251.910
       turn: action_jitter=0.126 action_fft_freq=1.106 joint_speed_mean=0.687 rod_speed_mean=0.029 contact_switch_hz=1.333 foot_slip=0.083 wz_active_frac=0.703 clock_restarts=0.000 swings_per_rad=30.759 slip_per_rad=2.040 work_per_rad=2188.571 cam_work_per_rad=703.954
  move_turn: action_jitter=0.106 action_fft_freq=1.041 joint_speed_mean=0.683 rod_speed_mean=0.028 contact_switch_hz=1.722 foot_slip=0.111 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=62.448 slip_per_rad=4.130 work_per_rad=4246.144 cam_work_per_rad=1611.102

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.277 steps_per_s=2.417 gait_hz=0.403 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=9.737 support_feet=3.062 taps_per_m=9.817 swing_apex=0.121 step_rate_hz=2.500 gait_hz=0.391 clock_hz=0.357 cadence_ratio=1.094 hz_speed_slope=-0.038 offset_R=0.172
       side: steps_per_m=10.485 support_feet=3.160 taps_per_m=5.097 swing_apex=0.100 step_rate_hz=2.500 gait_hz=0.394 clock_hz=0.314 cadence_ratio=1.253 hz_speed_slope=3.811 offset_R=0.045
       back: steps_per_m=12.210 support_feet=3.095 taps_per_m=10.546 swing_apex=0.109 step_rate_hz=2.750 gait_hz=0.415 clock_hz=0.314 cadence_ratio=1.321 hz_speed_slope=-0.455 offset_R=0.085
       diag: steps_per_m=11.425 support_feet=3.042 taps_per_m=7.577 swing_apex=0.092 step_rate_hz=2.917 gait_hz=0.464 clock_hz=0.357 cadence_ratio=1.290 hz_speed_slope=0.026 offset_R=0.123
       turn: steps_per_rad=19.181 support_feet=3.205 taps_per_rad=11.185 swing_apex=0.138 step_rate_hz=2.500 gait_hz=0.421 clock_hz=0.349 cadence_ratio=1.202 steps_per_rad_pre=13.337 steps_per_rad_post=5.740 offset_R=0.084
  move_turn: steps_per_rad=30.865 support_feet=3.076 taps_per_rad=30.365 swing_apex=0.109 step_rate_hz=2.667 gait_hz=0.366 clock_hz=0.500 cadence_ratio=0.732 hz_speed_slope=-0.064 offset_R=0.040

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
