```
=== crab-hex gait eval ===
scenario   : suite__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_03-27-53_seed19_X07_p3a_A15pB/model_22993.pt
episodes   : 1000  unscored: 7
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6188210295260517  p25=0.5027997523650583  p75=0.6980412159065453
tippy_tap_fraction median=0.43649167733674776
slip_ratio         median=0.21620034840520377
tracking_ratio     median=0.9947257075930286  (achieved/commanded vx, walking holds; n=699)
schedule_completion_rate=0.909
terminations={'schedule_complete': 909, 'fall': 91}
facing_error_rms   median=0.05903397818846412  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.24595203954926576 (n=309)
     stand: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.5241696864133509 (n=100)
       fwd: cmd 0.25 -> achieved 0.257 m/s | tripod median=0.5861272457901203 (n=100)
     lat_l: cmd 0.22 -> achieved 0.213 m/s | tripod median=0.6321657348495594 (n=95)
     lat_r: cmd 0.22 -> achieved 0.235 m/s | tripod median=0.7495652145491541 (n=100)
      back: cmd 0.22 -> achieved 0.233 m/s | tripod median=0.6144623884217755 (n=100)
      diag: cmd 0.25 -> achieved 0.246 m/s | tripod median=0.44212151144728573 (n=99)
  turn_l90: cmd 0.00 -> achieved -0.002 m/s | tripod median=0.7229310659085193 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.005 m/s | tripod median=0.7446239212518315 (n=100)
  move_turn: cmd 0.22 -> achieved 0.184 m/s | tripod median=0.39062616611705536 (n=98)
  move_turn_r: cmd 0.22 -> achieved 0.158 m/s | tripod median=0.6223676738433996 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.011 initial=0.006 final=0.010 max=0.021 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.003 yaw_rate_ratio=n/a
     stand: err rms=0.019 initial=0.010 final=0.018 max=0.046 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.056 initial=0.009 final=0.056 max=0.123 closed=n/a | settle=0.000 (n=97/100) | progress=-0.004 net=-0.001 yaw_rate_ratio=0.441
     lat_l: err rms=0.022 initial=0.010 final=0.018 max=0.050 closed=n/a | settle=0.000 (n=100/100) | progress=-0.023 net=0.001 yaw_rate_ratio=0.358
     lat_r: err rms=0.014 initial=0.010 final=0.011 max=0.035 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
      back: err rms=0.061 initial=0.008 final=0.078 max=0.134 closed=n/a | settle=0.000 (n=89/100) | progress=-0.002 net=-0.000 yaw_rate_ratio=0.419
      diag: err rms=0.048 initial=0.013 final=0.035 max=0.105 closed=n/a | settle=0.000 (n=90/100) | progress=-0.009 net=0.002 yaw_rate_ratio=0.487
  turn_l90: err rms=0.880 initial=1.567 final=0.141 max=1.571 closed=0.909 | settle=11.240 (n=56/100) | progress=0.125 net=0.119 yaw_rate_ratio=0.531
  turn_r90: err rms=0.874 initial=1.574 final=0.124 max=1.579 closed=0.922 | settle=11.180 (n=73/100) | progress=0.126 net=0.121 yaw_rate_ratio=0.536
  move_turn: err rms=1.233 initial=1.568 final=0.917 max=1.581 closed=0.418 | settle=n/a (n=0/100) | progress=0.060 net=0.061 yaw_rate_ratio=0.438
  move_turn_r: err rms=1.173 initial=1.574 final=0.825 max=1.577 closed=0.475 | settle=n/a (n=0/100) | progress=0.062 net=0.062 yaw_rate_ratio=0.445

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.002 tripod=0.524 | closed=n/a progress=n/a net=0.001 | drift=0.047 dir_err=n/a
        fwd: n=100 surv=0.940 (fwd=0.94) | ratio=1.026 achieved=0.257 tripod=0.586 | closed=n/a progress=-0.004 net=-0.001 | drift=0.258 dir_err=0.023
       side: n=200 surv=0.945 (lat_l=0.89 lat_r=1.00) | ratio=1.013 achieved=0.223 tripod=0.721 | closed=n/a progress=-0.023 net=0.001 | drift=0.227 dir_err=0.013
       back: n=100 surv=1.000 (back=1.00) | ratio=1.057 achieved=0.233 tripod=0.614 | closed=n/a progress=-0.002 net=-0.000 | drift=0.234 dir_err=0.030
       diag: n=100 surv=0.660 (diag=0.66) | ratio=0.983 achieved=0.246 tripod=0.442 | closed=n/a progress=-0.009 net=0.002 | drift=0.252 dir_err=0.039
       turn: n=200 surv=1.000 (turn_l90=1.00 turn_r90=1.00) | ratio=n/a achieved=0.001 tripod=0.734 | closed=0.914 progress=0.125 net=0.119 | drift=0.048 dir_err=n/a
  move_turn: n=200 surv=0.805 (move_turn=0.63 move_turn_r=0.98) | ratio=0.764 achieved=0.168 tripod=0.551 | closed=0.455 progress=0.061 net=0.062 | drift=0.184 dir_err=0.058

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.167 action_fft_freq=1.670 joint_speed_mean=0.345 rod_speed_mean=0.029 contact_switch_hz=2.097 foot_slip=0.093 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.005 settle_six_down_s=12.000 foot_disp_end=0.231 work_per_s=331.802
        fwd: action_jitter=0.129 action_fft_freq=1.127 joint_speed_mean=0.725 rod_speed_mean=0.029 contact_switch_hz=1.792 foot_slip=0.118 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=21.000 slip_per_m=1.413 work_per_m=1436.144 cam_work_per_m=593.186
       side: action_jitter=0.094 action_fft_freq=1.075 joint_speed_mean=0.350 rod_speed_mean=0.028 contact_switch_hz=1.243 foot_slip=0.092 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=17.131 slip_per_m=1.347 work_per_m=1470.412 cam_work_per_m=213.928
       back: action_jitter=0.130 action_fft_freq=1.130 joint_speed_mean=0.705 rod_speed_mean=0.028 contact_switch_hz=1.785 foot_slip=0.115 wz_active_frac=0.005 clock_restarts=0.000 swings_per_m=23.145 slip_per_m=1.525 work_per_m=1466.767 cam_work_per_m=624.027
       diag: action_jitter=0.116 action_fft_freq=1.189 joint_speed_mean=0.615 rod_speed_mean=0.028 contact_switch_hz=2.064 foot_slip=0.174 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=25.652 slip_per_m=2.569 work_per_m=1624.097 cam_work_per_m=625.389
       turn: action_jitter=0.125 action_fft_freq=1.074 joint_speed_mean=0.689 rod_speed_mean=0.029 contact_switch_hz=1.396 foot_slip=0.096 wz_active_frac=0.947 clock_restarts=1.000 swings_per_rad=35.275 slip_per_rad=2.622 work_per_rad=2465.020 cam_work_per_rad=847.399
  move_turn: action_jitter=0.119 action_fft_freq=1.105 joint_speed_mean=0.645 rod_speed_mean=0.028 contact_switch_hz=1.625 foot_slip=0.127 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=77.572 slip_per_rad=6.400 work_per_rad=6204.820 cam_work_per_rad=2656.402

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.334 steps_per_s=2.875 gait_hz=0.391 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=10.565 support_feet=3.088 taps_per_m=10.125 swing_apex=0.102 step_rate_hz=2.667 gait_hz=0.391 clock_hz=0.357 cadence_ratio=1.078 hz_speed_slope=-0.873 offset_R=0.093
       side: steps_per_m=10.555 support_feet=3.220 taps_per_m=6.174 swing_apex=0.102 step_rate_hz=2.333 gait_hz=0.354 clock_hz=0.314 cadence_ratio=1.126 hz_speed_slope=0.660 offset_R=0.095
       back: steps_per_m=12.094 support_feet=3.065 taps_per_m=10.878 swing_apex=0.102 step_rate_hz=2.750 gait_hz=0.391 clock_hz=0.314 cadence_ratio=1.223 hz_speed_slope=0.632 offset_R=0.109
       diag: steps_per_m=12.262 support_feet=3.383 taps_per_m=12.969 swing_apex=0.075 step_rate_hz=2.922 gait_hz=0.452 clock_hz=0.357 cadence_ratio=1.257 hz_speed_slope=-0.749 offset_R=0.134
       turn: steps_per_rad=21.457 support_feet=3.241 taps_per_rad=13.481 swing_apex=0.114 step_rate_hz=2.583 gait_hz=0.403 clock_hz=0.467 cadence_ratio=0.864 steps_per_rad_pre=19.939 steps_per_rad_post=2.001 offset_R=0.081
  move_turn: steps_per_rad=41.790 support_feet=3.107 taps_per_rad=36.251 swing_apex=0.109 step_rate_hz=2.583 gait_hz=0.372 clock_hz=0.500 cadence_ratio=0.745 hz_speed_slope=1.228 offset_R=0.036

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
