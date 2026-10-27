```
=== crab-hex gait eval ===
scenario   : suite__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_30000.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6941642612192716  p25=0.5800980825903209  p75=0.7661429516379095
tippy_tap_fraction median=0.36363636363636365
slip_ratio         median=0.20070829505046925
tracking_ratio     median=0.996128588313101  (achieved/commanded vx, walking holds; n=700)
schedule_completion_rate=0.972
terminations={'schedule_complete': 972, 'fall': 28}
facing_error_rms   median=0.05819333047664778  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved -0.002 m/s | tripod median=0.7152486340690857 (n=997)
     stand: cmd 0.00 -> achieved 0.000 m/s | tripod median=0.7669918192971419 (n=100)
       fwd: cmd 0.25 -> achieved 0.244 m/s | tripod median=0.5357914586155734 (n=100)
     lat_l: cmd 0.22 -> achieved 0.226 m/s | tripod median=0.5868945344640846 (n=100)
     lat_r: cmd 0.22 -> achieved 0.227 m/s | tripod median=0.7703968262193885 (n=100)
      back: cmd 0.22 -> achieved 0.220 m/s | tripod median=0.7547165536525033 (n=100)
      diag: cmd 0.25 -> achieved 0.255 m/s | tripod median=0.5736846515160733 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.006 m/s | tripod median=0.7786974090465117 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.787598818012806 (n=100)
  move_turn: cmd 0.22 -> achieved 0.156 m/s | tripod median=0.3477106337592908 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.181 m/s | tripod median=0.647614747060089 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.022 initial=0.017 final=0.026 max=0.036 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=-0.005 yaw_rate_ratio=n/a
     stand: err rms=0.035 initial=0.029 final=0.043 max=0.064 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.047 initial=0.028 final=0.035 max=0.126 closed=n/a | settle=0.000 (n=97/100) | progress=-0.000 net=0.001 yaw_rate_ratio=0.749
     lat_l: err rms=0.024 initial=0.026 final=0.017 max=0.056 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
     lat_r: err rms=0.042 initial=0.027 final=0.050 max=0.072 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.002 yaw_rate_ratio=n/a
      back: err rms=0.071 initial=0.028 final=0.110 max=0.138 closed=n/a | settle=0.000 (n=98/100) | progress=-0.002 net=-0.007 yaw_rate_ratio=0.294
      diag: err rms=0.036 initial=0.024 final=0.020 max=0.092 closed=n/a | settle=0.000 (n=100/100) | progress=0.009 net=0.003 yaw_rate_ratio=0.404
  turn_l90: err rms=0.780 initial=1.544 final=0.016 max=1.546 closed=0.990 | settle=8.170 (n=100/100) | progress=0.171 net=0.128 yaw_rate_ratio=0.622
  turn_r90: err rms=0.784 initial=1.596 final=0.080 max=1.599 closed=0.950 | settle=8.080 (n=100/100) | progress=0.180 net=0.127 yaw_rate_ratio=0.659
  move_turn: err rms=1.150 initial=1.545 final=0.897 max=1.546 closed=0.421 | settle=n/a (n=0/100) | progress=0.055 net=0.057 yaw_rate_ratio=0.445
  move_turn_r: err rms=1.021 initial=1.596 final=0.241 max=1.602 closed=0.848 | settle=n/a (n=0/100) | progress=0.113 net=0.113 yaw_rate_ratio=0.478

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.000 tripod=0.767 | closed=n/a progress=n/a net=-0.001 | drift=0.032 dir_err=n/a
        fwd: n=100 surv=1.000 (fwd=1.00) | ratio=0.975 achieved=0.244 tripod=0.536 | closed=n/a progress=-0.000 net=0.001 | drift=0.246 dir_err=0.030
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=1.027 achieved=0.226 tripod=0.684 | closed=n/a progress=n/a net=-0.000 | drift=0.227 dir_err=0.016
       back: n=100 surv=1.000 (back=1.00) | ratio=0.998 achieved=0.220 tripod=0.755 | closed=n/a progress=-0.002 net=-0.007 | drift=0.220 dir_err=0.050
       diag: n=100 surv=1.000 (diag=1.00) | ratio=1.019 achieved=0.255 tripod=0.574 | closed=n/a progress=0.009 net=0.003 | drift=0.256 dir_err=0.098
       turn: n=200 surv=1.000 (turn_l90=1.00 turn_r90=1.00) | ratio=n/a achieved=0.000 tripod=0.784 | closed=0.972 progress=0.174 net=0.128 | drift=0.046 dir_err=n/a
  move_turn: n=200 surv=0.860 (move_turn=0.72 move_turn_r=1.00) | ratio=0.781 achieved=0.172 tripod=0.510 | closed=0.608 progress=0.083 net=0.085 | drift=0.185 dir_err=0.072

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.118 action_fft_freq=1.420 joint_speed_mean=0.367 rod_speed_mean=0.029 contact_switch_hz=1.361 foot_slip=0.071 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.027 settle_six_down_s=12.000 foot_disp_end=0.099 work_per_s=325.067
        fwd: action_jitter=0.090 action_fft_freq=0.973 joint_speed_mean=0.705 rod_speed_mean=0.029 contact_switch_hz=1.736 foot_slip=0.111 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=21.327 slip_per_m=1.390 work_per_m=1430.636 cam_work_per_m=639.140
       side: action_jitter=0.070 action_fft_freq=0.940 joint_speed_mean=0.362 rod_speed_mean=0.028 contact_switch_hz=1.201 foot_slip=0.090 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=15.962 slip_per_m=1.275 work_per_m=1336.041 cam_work_per_m=260.286
       back: action_jitter=0.109 action_fft_freq=1.083 joint_speed_mean=0.685 rod_speed_mean=0.029 contact_switch_hz=1.479 foot_slip=0.094 wz_active_frac=0.012 clock_restarts=0.000 swings_per_m=20.491 slip_per_m=1.355 work_per_m=1408.747 cam_work_per_m=441.473
       diag: action_jitter=0.084 action_fft_freq=1.066 joint_speed_mean=0.525 rod_speed_mean=0.028 contact_switch_hz=1.861 foot_slip=0.108 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=22.023 slip_per_m=1.326 work_per_m=1283.276 cam_work_per_m=262.277
       turn: action_jitter=0.099 action_fft_freq=1.042 joint_speed_mean=0.667 rod_speed_mean=0.029 contact_switch_hz=1.278 foot_slip=0.081 wz_active_frac=0.685 clock_restarts=0.000 swings_per_rad=30.317 slip_per_rad=2.044 work_per_rad=2364.979 cam_work_per_rad=699.620
  move_turn: action_jitter=0.080 action_fft_freq=0.951 joint_speed_mean=0.654 rod_speed_mean=0.028 contact_switch_hz=1.667 foot_slip=0.114 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=58.466 slip_per_rad=4.155 work_per_rad=4185.128 cam_work_per_rad=1927.094

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.288 steps_per_s=2.833 gait_hz=0.482 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=11.922 support_feet=3.048 taps_per_m=9.468 swing_apex=0.110 step_rate_hz=2.917 gait_hz=0.409 clock_hz=0.357 cadence_ratio=1.134 hz_speed_slope=-0.257 offset_R=0.081
       side: steps_per_m=10.509 support_feet=3.188 taps_per_m=5.736 swing_apex=0.114 step_rate_hz=2.333 gait_hz=0.369 clock_hz=0.314 cadence_ratio=1.175 hz_speed_slope=-1.738 offset_R=0.056
       back: steps_per_m=12.175 support_feet=3.162 taps_per_m=7.946 swing_apex=0.117 step_rate_hz=2.667 gait_hz=0.433 clock_hz=0.315 cadence_ratio=1.379 hz_speed_slope=0.224 offset_R=0.076
       diag: steps_per_m=12.226 support_feet=3.109 taps_per_m=9.545 swing_apex=0.080 step_rate_hz=3.083 gait_hz=0.464 clock_hz=0.357 cadence_ratio=1.299 hz_speed_slope=0.283 offset_R=0.095
       turn: steps_per_rad=19.981 support_feet=3.225 taps_per_rad=9.780 swing_apex=0.143 step_rate_hz=2.500 gait_hz=0.427 clock_hz=0.341 cadence_ratio=1.247 steps_per_rad_pre=13.815 steps_per_rad_post=5.918 offset_R=0.034
  move_turn: steps_per_rad=32.530 support_feet=3.027 taps_per_rad=25.968 swing_apex=0.095 step_rate_hz=2.750 gait_hz=0.385 clock_hz=0.500 cadence_ratio=0.769 hz_speed_slope=-0.369 offset_R=0.045

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
