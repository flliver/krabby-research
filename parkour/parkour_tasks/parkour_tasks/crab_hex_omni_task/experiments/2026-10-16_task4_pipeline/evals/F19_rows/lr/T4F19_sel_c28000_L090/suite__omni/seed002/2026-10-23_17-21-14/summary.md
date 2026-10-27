```
=== crab-hex gait eval ===
scenario   : suite__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_28000.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6881434403432469  p25=0.6466801204622121  p75=0.7332201989475597
tippy_tap_fraction median=0.3706690561529271
slip_ratio         median=0.19991946460746865
tracking_ratio     median=1.0143009549716075  (achieved/commanded vx, walking holds; n=700)
schedule_completion_rate=1.0
terminations={'schedule_complete': 1000}
facing_error_rms   median=0.05848925187709894  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.5991707267290894 (n=933)
     stand: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7426693906701777 (n=100)
       fwd: cmd 0.25 -> achieved 0.253 m/s | tripod median=0.6672977113017402 (n=100)
     lat_l: cmd 0.22 -> achieved 0.230 m/s | tripod median=0.6607698521714791 (n=100)
     lat_r: cmd 0.22 -> achieved 0.232 m/s | tripod median=0.7494291708675156 (n=100)
      back: cmd 0.22 -> achieved 0.221 m/s | tripod median=0.7000475295393687 (n=100)
      diag: cmd 0.25 -> achieved 0.257 m/s | tripod median=0.6796180660832922 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.008 m/s | tripod median=0.7805112450324112 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.008 m/s | tripod median=0.7763916381153703 (n=100)
  move_turn: cmd 0.22 -> achieved 0.170 m/s | tripod median=0.5824252681518162 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.169 m/s | tripod median=0.6714293742312971 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.012 initial=0.008 final=0.016 max=0.027 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.000 yaw_rate_ratio=n/a
     stand: err rms=0.032 initial=0.015 final=0.039 max=0.060 closed=n/a | settle=0.000 (n=100/100) | progress=0.000 net=-0.001 yaw_rate_ratio=0.284
       fwd: err rms=0.042 initial=0.013 final=0.045 max=0.105 closed=n/a | settle=0.000 (n=99/100) | progress=-0.004 net=0.003 yaw_rate_ratio=0.808
     lat_l: err rms=0.022 initial=0.015 final=0.022 max=0.053 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.002 yaw_rate_ratio=n/a
     lat_r: err rms=0.028 initial=0.014 final=0.028 max=0.062 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
      back: err rms=0.080 initial=0.016 final=0.087 max=0.150 closed=n/a | settle=2.090 (n=100/100) | progress=-0.001 net=-0.004 yaw_rate_ratio=0.646
      diag: err rms=0.038 initial=0.016 final=0.052 max=0.085 closed=n/a | settle=0.000 (n=100/100) | progress=-0.003 net=0.003 yaw_rate_ratio=0.344
  turn_l90: err rms=0.790 initial=1.565 final=0.015 max=1.572 closed=0.990 | settle=8.400 (n=100/100) | progress=0.170 net=0.131 yaw_rate_ratio=0.659
  turn_r90: err rms=0.817 initial=1.578 final=0.026 max=1.583 closed=0.984 | settle=8.630 (n=100/100) | progress=0.167 net=0.130 yaw_rate_ratio=0.645
  move_turn: err rms=1.136 initial=1.560 final=0.753 max=1.572 closed=0.519 | settle=n/a (n=0/100) | progress=0.066 net=0.067 yaw_rate_ratio=0.432
  move_turn_r: err rms=1.018 initial=1.580 final=0.359 max=1.587 closed=0.774 | settle=11.640 (n=3/100) | progress=0.102 net=0.102 yaw_rate_ratio=0.469

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.003 tripod=0.743 | closed=n/a progress=0.000 net=-0.001 | drift=0.030 dir_err=n/a
        fwd: n=100 surv=1.000 (fwd=1.00) | ratio=1.013 achieved=0.253 tripod=0.667 | closed=n/a progress=-0.004 net=0.003 | drift=0.253 dir_err=0.019
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=1.050 achieved=0.231 tripod=0.703 | closed=n/a progress=n/a net=0.002 | drift=0.232 dir_err=0.018
       back: n=100 surv=1.000 (back=1.00) | ratio=1.006 achieved=0.221 tripod=0.700 | closed=n/a progress=-0.001 net=-0.004 | drift=0.222 dir_err=0.022
       diag: n=100 surv=1.000 (diag=1.00) | ratio=1.028 achieved=0.257 tripod=0.680 | closed=n/a progress=-0.003 net=0.003 | drift=0.260 dir_err=0.049
       turn: n=200 surv=1.000 (turn_l90=1.00 turn_r90=1.00) | ratio=n/a achieved=0.001 tripod=0.778 | closed=0.988 progress=0.168 net=0.131 | drift=0.047 dir_err=n/a
  move_turn: n=200 surv=1.000 (move_turn=1.00 move_turn_r=1.00) | ratio=0.772 achieved=0.170 tripod=0.619 | closed=0.634 progress=0.083 net=0.084 | drift=0.182 dir_err=0.126

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.123 action_fft_freq=1.386 joint_speed_mean=0.357 rod_speed_mean=0.028 contact_switch_hz=1.194 foot_slip=0.082 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.023 settle_six_down_s=12.000 foot_disp_end=0.112 work_per_s=372.470
        fwd: action_jitter=0.092 action_fft_freq=0.958 joint_speed_mean=0.757 rod_speed_mean=0.029 contact_switch_hz=1.611 foot_slip=0.104 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=19.321 slip_per_m=1.280 work_per_m=1353.056 cam_work_per_m=534.840
       side: action_jitter=0.072 action_fft_freq=0.993 joint_speed_mean=0.385 rod_speed_mean=0.028 contact_switch_hz=1.236 foot_slip=0.085 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=16.215 slip_per_m=1.189 work_per_m=1297.354 cam_work_per_m=236.435
       back: action_jitter=0.100 action_fft_freq=1.023 joint_speed_mean=0.706 rod_speed_mean=0.029 contact_switch_hz=1.611 foot_slip=0.104 wz_active_frac=0.030 clock_restarts=0.000 swings_per_m=22.403 slip_per_m=1.487 work_per_m=1464.146 cam_work_per_m=482.977
       diag: action_jitter=0.086 action_fft_freq=1.074 joint_speed_mean=0.565 rod_speed_mean=0.029 contact_switch_hz=1.576 foot_slip=0.100 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=18.475 slip_per_m=1.185 work_per_m=1205.660 cam_work_per_m=247.869
       turn: action_jitter=0.097 action_fft_freq=0.995 joint_speed_mean=0.689 rod_speed_mean=0.029 contact_switch_hz=1.250 foot_slip=0.083 wz_active_frac=0.713 clock_restarts=0.000 swings_per_rad=28.818 slip_per_rad=2.048 work_per_rad=2251.471 cam_work_per_rad=753.277
  move_turn: action_jitter=0.084 action_fft_freq=0.968 joint_speed_mean=0.677 rod_speed_mean=0.028 contact_switch_hz=1.653 foot_slip=0.112 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=59.162 slip_per_rad=4.078 work_per_rad=4111.338 cam_work_per_rad=1623.762

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.337 steps_per_s=2.167 gait_hz=0.366 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=10.335 support_feet=3.085 taps_per_m=8.774 swing_apex=0.121 step_rate_hz=2.583 gait_hz=0.391 clock_hz=0.357 cadence_ratio=1.094 hz_speed_slope=-0.089 offset_R=0.129
       side: steps_per_m=10.831 support_feet=3.152 taps_per_m=5.071 swing_apex=0.102 step_rate_hz=2.500 gait_hz=0.391 clock_hz=0.314 cadence_ratio=1.243 hz_speed_slope=0.422 offset_R=0.011
       back: steps_per_m=12.290 support_feet=3.109 taps_per_m=9.973 swing_apex=0.108 step_rate_hz=2.667 gait_hz=0.409 clock_hz=0.316 cadence_ratio=1.295 hz_speed_slope=0.994 offset_R=0.109
       diag: steps_per_m=11.730 support_feet=3.046 taps_per_m=6.500 swing_apex=0.089 step_rate_hz=3.000 gait_hz=0.446 clock_hz=0.357 cadence_ratio=1.248 hz_speed_slope=0.351 offset_R=0.215
       turn: steps_per_rad=19.370 support_feet=3.221 taps_per_rad=9.338 swing_apex=0.137 step_rate_hz=2.500 gait_hz=0.421 clock_hz=0.355 cadence_ratio=1.193 steps_per_rad_pre=13.991 steps_per_rad_post=5.649 offset_R=0.092
  move_turn: steps_per_rad=32.861 support_feet=3.072 taps_per_rad=25.881 swing_apex=0.104 step_rate_hz=2.708 gait_hz=0.366 clock_hz=0.500 cadence_ratio=0.732 hz_speed_slope=-0.311 offset_R=0.078

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
