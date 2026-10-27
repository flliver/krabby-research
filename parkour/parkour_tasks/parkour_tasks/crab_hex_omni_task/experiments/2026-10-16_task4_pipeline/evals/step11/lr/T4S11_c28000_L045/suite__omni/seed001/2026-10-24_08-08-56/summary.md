```
=== crab-hex gait eval ===
scenario   : suite__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_28000.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7006094301523365  p25=0.6611475547894736  p75=0.7458284362905223
tippy_tap_fraction median=0.390625
slip_ratio         median=0.20521914934261715
tracking_ratio     median=1.0066219576889655  (achieved/commanded vx, walking holds; n=700)
schedule_completion_rate=1.0
terminations={'schedule_complete': 1000}
facing_error_rms   median=0.05948276139646573  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.6419087521302982 (n=949)
     stand: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7490372770248432 (n=100)
       fwd: cmd 0.25 -> achieved 0.251 m/s | tripod median=0.6780955707018699 (n=100)
     lat_l: cmd 0.22 -> achieved 0.231 m/s | tripod median=0.6646895776292242 (n=100)
     lat_r: cmd 0.22 -> achieved 0.232 m/s | tripod median=0.7514518914810404 (n=100)
      back: cmd 0.22 -> achieved 0.218 m/s | tripod median=0.7072938439699101 (n=100)
      diag: cmd 0.25 -> achieved 0.255 m/s | tripod median=0.6892507329651154 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.007 m/s | tripod median=0.7906176213780711 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.007 m/s | tripod median=0.7916487510809835 (n=100)
  move_turn: cmd 0.22 -> achieved 0.166 m/s | tripod median=0.6137388622099568 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.166 m/s | tripod median=0.6993116728344888 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.014 initial=0.008 final=0.016 max=0.030 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.000 yaw_rate_ratio=n/a
     stand: err rms=0.031 initial=0.016 final=0.037 max=0.066 closed=n/a | settle=0.000 (n=100/100) | progress=0.005 net=-0.001 yaw_rate_ratio=0.347
       fwd: err rms=0.048 initial=0.018 final=0.043 max=0.118 closed=n/a | settle=0.000 (n=100/100) | progress=-0.003 net=0.003 yaw_rate_ratio=0.603
     lat_l: err rms=0.020 initial=0.014 final=0.020 max=0.048 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.002 yaw_rate_ratio=n/a
     lat_r: err rms=0.029 initial=0.016 final=0.029 max=0.063 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
      back: err rms=0.084 initial=0.017 final=0.095 max=0.151 closed=n/a | settle=4.880 (n=100/100) | progress=-0.001 net=-0.004 yaw_rate_ratio=0.554
      diag: err rms=0.036 initial=0.016 final=0.056 max=0.086 closed=n/a | settle=0.000 (n=100/100) | progress=-0.007 net=0.003 yaw_rate_ratio=0.307
  turn_l90: err rms=0.780 initial=1.561 final=0.022 max=1.565 closed=0.986 | settle=8.260 (n=100/100) | progress=0.171 net=0.131 yaw_rate_ratio=0.660
  turn_r90: err rms=0.809 initial=1.581 final=0.023 max=1.588 closed=0.986 | settle=8.500 (n=100/100) | progress=0.169 net=0.131 yaw_rate_ratio=0.647
  move_turn: err rms=1.119 initial=1.561 final=0.688 max=1.571 closed=0.559 | settle=n/a (n=0/100) | progress=0.072 net=0.073 yaw_rate_ratio=0.416
  move_turn_r: err rms=1.010 initial=1.580 final=0.330 max=1.585 closed=0.790 | settle=11.960 (n=1/100) | progress=0.103 net=0.103 yaw_rate_ratio=0.464

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.003 tripod=0.749 | closed=n/a progress=0.005 net=-0.001 | drift=0.032 dir_err=n/a
        fwd: n=100 surv=1.000 (fwd=1.00) | ratio=1.005 achieved=0.251 tripod=0.678 | closed=n/a progress=-0.003 net=0.003 | drift=0.251 dir_err=0.020
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=1.051 achieved=0.231 tripod=0.715 | closed=n/a progress=n/a net=0.002 | drift=0.233 dir_err=0.017
       back: n=100 surv=1.000 (back=1.00) | ratio=0.993 achieved=0.218 tripod=0.707 | closed=n/a progress=-0.001 net=-0.004 | drift=0.219 dir_err=0.026
       diag: n=100 surv=1.000 (diag=1.00) | ratio=1.021 achieved=0.255 tripod=0.689 | closed=n/a progress=-0.007 net=0.003 | drift=0.259 dir_err=0.055
       turn: n=200 surv=1.000 (turn_l90=1.00 turn_r90=1.00) | ratio=n/a achieved=0.001 tripod=0.792 | closed=0.986 progress=0.170 net=0.131 | drift=0.046 dir_err=n/a
  move_turn: n=200 surv=1.000 (move_turn=1.00 move_turn_r=1.00) | ratio=0.755 achieved=0.166 tripod=0.659 | closed=0.666 progress=0.087 net=0.088 | drift=0.179 dir_err=0.148

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.134 action_fft_freq=1.506 joint_speed_mean=0.360 rod_speed_mean=0.028 contact_switch_hz=1.396 foot_slip=0.078 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.018 settle_six_down_s=12.000 foot_disp_end=0.118 work_per_s=335.836
        fwd: action_jitter=0.099 action_fft_freq=0.997 joint_speed_mean=0.756 rod_speed_mean=0.029 contact_switch_hz=1.667 foot_slip=0.100 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=20.144 slip_per_m=1.213 work_per_m=1235.957 cam_work_per_m=479.646
       side: action_jitter=0.076 action_fft_freq=1.047 joint_speed_mean=0.383 rod_speed_mean=0.029 contact_switch_hz=1.250 foot_slip=0.079 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=16.452 slip_per_m=1.070 work_per_m=1167.570 cam_work_per_m=208.896
       back: action_jitter=0.106 action_fft_freq=1.074 joint_speed_mean=0.705 rod_speed_mean=0.029 contact_switch_hz=1.674 foot_slip=0.099 wz_active_frac=0.030 clock_restarts=0.000 swings_per_m=23.438 slip_per_m=1.384 work_per_m=1337.755 cam_work_per_m=433.149
       diag: action_jitter=0.092 action_fft_freq=1.130 joint_speed_mean=0.568 rod_speed_mean=0.029 contact_switch_hz=1.625 foot_slip=0.093 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=19.156 slip_per_m=1.087 work_per_m=1114.490 cam_work_per_m=226.783
       turn: action_jitter=0.103 action_fft_freq=1.050 joint_speed_mean=0.684 rod_speed_mean=0.029 contact_switch_hz=1.361 foot_slip=0.078 wz_active_frac=0.705 clock_restarts=0.000 swings_per_rad=31.210 slip_per_rad=1.895 work_per_rad=2084.323 cam_work_per_rad=681.132
  move_turn: action_jitter=0.088 action_fft_freq=0.994 joint_speed_mean=0.682 rod_speed_mean=0.028 contact_switch_hz=1.722 foot_slip=0.107 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=59.702 slip_per_rad=3.615 work_per_rad=3636.006 cam_work_per_rad=1343.527

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.283 steps_per_s=2.250 gait_hz=0.378 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=10.340 support_feet=3.030 taps_per_m=9.584 swing_apex=0.119 step_rate_hz=2.583 gait_hz=0.397 clock_hz=0.357 cadence_ratio=1.110 hz_speed_slope=-0.288 offset_R=0.112
       side: steps_per_m=10.944 support_feet=3.108 taps_per_m=5.860 swing_apex=0.103 step_rate_hz=2.500 gait_hz=0.397 clock_hz=0.314 cadence_ratio=1.262 hz_speed_slope=0.626 offset_R=0.057
       back: steps_per_m=12.672 support_feet=3.033 taps_per_m=10.592 swing_apex=0.109 step_rate_hz=2.750 gait_hz=0.415 clock_hz=0.316 cadence_ratio=1.312 hz_speed_slope=-0.213 offset_R=0.033
       diag: steps_per_m=11.945 support_feet=2.995 taps_per_m=6.870 swing_apex=0.092 step_rate_hz=3.083 gait_hz=0.458 clock_hz=0.357 cadence_ratio=1.282 hz_speed_slope=-0.133 offset_R=0.071
       turn: steps_per_rad=19.488 support_feet=3.167 taps_per_rad=11.685 swing_apex=0.139 step_rate_hz=2.583 gait_hz=0.433 clock_hz=0.351 cadence_ratio=1.231 steps_per_rad_pre=13.824 steps_per_rad_post=5.723 offset_R=0.020
  move_turn: steps_per_rad=31.314 support_feet=3.031 taps_per_rad=28.849 swing_apex=0.108 step_rate_hz=2.667 gait_hz=0.366 clock_hz=0.500 cadence_ratio=0.732 hz_speed_slope=0.093 offset_R=0.017

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
