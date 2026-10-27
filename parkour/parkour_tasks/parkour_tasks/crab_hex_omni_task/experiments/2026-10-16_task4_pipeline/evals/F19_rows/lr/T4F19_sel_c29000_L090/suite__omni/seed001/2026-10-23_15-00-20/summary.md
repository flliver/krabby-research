```
=== crab-hex gait eval ===
scenario   : suite__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_29000.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6847351125799643  p25=0.6402624465960403  p75=0.742920612632471
tippy_tap_fraction median=0.36923076923076925
slip_ratio         median=0.19718656549423297
tracking_ratio     median=1.016329019960673  (achieved/commanded vx, walking holds; n=700)
schedule_completion_rate=1.0
terminations={'schedule_complete': 1000}
facing_error_rms   median=0.053051168058120546  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.005 m/s | tripod median=0.6407036565114375 (n=994)
     stand: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7439872530151199 (n=100)
       fwd: cmd 0.25 -> achieved 0.256 m/s | tripod median=0.6485341445293042 (n=100)
     lat_l: cmd 0.22 -> achieved 0.229 m/s | tripod median=0.6443027063882502 (n=100)
     lat_r: cmd 0.22 -> achieved 0.232 m/s | tripod median=0.7630118253538314 (n=100)
      back: cmd 0.22 -> achieved 0.222 m/s | tripod median=0.6948584476836653 (n=100)
      diag: cmd 0.25 -> achieved 0.256 m/s | tripod median=0.67757741533949 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.007 m/s | tripod median=0.7745787358571875 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.007 m/s | tripod median=0.774553959138334 (n=100)
  move_turn: cmd 0.22 -> achieved 0.174 m/s | tripod median=0.5722673180785428 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.170 m/s | tripod median=0.6586211101038091 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.011 initial=0.007 final=0.014 max=0.026 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.001 yaw_rate_ratio=n/a
     stand: err rms=0.028 initial=0.011 final=0.033 max=0.059 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a
       fwd: err rms=0.037 initial=0.015 final=0.030 max=0.096 closed=n/a | settle=0.000 (n=100/100) | progress=0.003 net=0.002 yaw_rate_ratio=1.025
     lat_l: err rms=0.026 initial=0.013 final=0.029 max=0.060 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.002 yaw_rate_ratio=n/a
     lat_r: err rms=0.027 initial=0.015 final=0.029 max=0.060 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.002 yaw_rate_ratio=n/a
      back: err rms=0.075 initial=0.012 final=0.093 max=0.138 closed=n/a | settle=0.000 (n=98/100) | progress=-0.001 net=-0.003 yaw_rate_ratio=0.554
      diag: err rms=0.036 initial=0.015 final=0.045 max=0.084 closed=n/a | settle=0.000 (n=100/100) | progress=-0.001 net=0.001 yaw_rate_ratio=0.375
  turn_l90: err rms=0.784 initial=1.567 final=0.017 max=1.570 closed=0.989 | settle=8.320 (n=100/100) | progress=0.171 net=0.131 yaw_rate_ratio=0.655
  turn_r90: err rms=0.807 initial=1.575 final=0.023 max=1.583 closed=0.986 | settle=8.580 (n=100/100) | progress=0.168 net=0.130 yaw_rate_ratio=0.653
  move_turn: err rms=1.149 initial=1.566 final=0.769 max=1.573 closed=0.508 | settle=n/a (n=0/100) | progress=0.065 net=0.066 yaw_rate_ratio=0.432
  move_turn_r: err rms=1.029 initial=1.578 final=0.385 max=1.585 closed=0.754 | settle=n/a (n=0/100) | progress=0.099 net=0.099 yaw_rate_ratio=0.461

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.003 tripod=0.744 | closed=n/a progress=n/a net=0.000 | drift=0.030 dir_err=n/a
        fwd: n=100 surv=1.000 (fwd=1.00) | ratio=1.025 achieved=0.256 tripod=0.649 | closed=n/a progress=0.003 net=0.002 | drift=0.256 dir_err=0.016
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=1.048 achieved=0.231 tripod=0.704 | closed=n/a progress=n/a net=0.002 | drift=0.232 dir_err=0.024
       back: n=100 surv=1.000 (back=1.00) | ratio=1.008 achieved=0.222 tripod=0.695 | closed=n/a progress=-0.001 net=-0.003 | drift=0.222 dir_err=0.010
       diag: n=100 surv=1.000 (diag=1.00) | ratio=1.024 achieved=0.256 tripod=0.678 | closed=n/a progress=-0.001 net=0.001 | drift=0.258 dir_err=0.040
       turn: n=200 surv=1.000 (turn_l90=1.00 turn_r90=1.00) | ratio=n/a achieved=0.001 tripod=0.775 | closed=0.988 progress=0.169 net=0.131 | drift=0.047 dir_err=n/a
  move_turn: n=200 surv=1.000 (move_turn=1.00 move_turn_r=1.00) | ratio=0.782 achieved=0.172 tripod=0.617 | closed=0.629 progress=0.082 net=0.083 | drift=0.184 dir_err=0.114

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.124 action_fft_freq=1.417 joint_speed_mean=0.364 rod_speed_mean=0.028 contact_switch_hz=1.250 foot_slip=0.077 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.023 settle_six_down_s=12.000 foot_disp_end=0.112 work_per_s=361.933
        fwd: action_jitter=0.089 action_fft_freq=0.952 joint_speed_mean=0.759 rod_speed_mean=0.029 contact_switch_hz=1.611 foot_slip=0.109 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=18.980 slip_per_m=1.324 work_per_m=1358.488 cam_work_per_m=548.148
       side: action_jitter=0.072 action_fft_freq=1.005 joint_speed_mean=0.387 rod_speed_mean=0.028 contact_switch_hz=1.208 foot_slip=0.084 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=15.978 slip_per_m=1.156 work_per_m=1284.923 cam_work_per_m=229.931
       back: action_jitter=0.101 action_fft_freq=1.021 joint_speed_mean=0.701 rod_speed_mean=0.029 contact_switch_hz=1.604 foot_slip=0.102 wz_active_frac=0.010 clock_restarts=0.000 swings_per_m=21.979 slip_per_m=1.460 work_per_m=1443.852 cam_work_per_m=502.784
       diag: action_jitter=0.088 action_fft_freq=1.067 joint_speed_mean=0.572 rod_speed_mean=0.029 contact_switch_hz=1.597 foot_slip=0.099 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=18.824 slip_per_m=1.185 work_per_m=1193.203 cam_work_per_m=245.669
       turn: action_jitter=0.096 action_fft_freq=0.993 joint_speed_mean=0.687 rod_speed_mean=0.029 contact_switch_hz=1.222 foot_slip=0.083 wz_active_frac=0.710 clock_restarts=0.000 swings_per_rad=28.494 slip_per_rad=2.049 work_per_rad=2243.514 cam_work_per_rad=755.854
  move_turn: action_jitter=0.085 action_fft_freq=0.973 joint_speed_mean=0.678 rod_speed_mean=0.028 contact_switch_hz=1.639 foot_slip=0.111 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=59.153 slip_per_rad=4.283 work_per_rad=4380.342 cam_work_per_rad=1683.429

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.319 steps_per_s=2.333 gait_hz=0.397 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=10.132 support_feet=3.102 taps_per_m=8.709 swing_apex=0.118 step_rate_hz=2.583 gait_hz=0.391 clock_hz=0.357 cadence_ratio=1.094 hz_speed_slope=-0.292 offset_R=0.065
       side: steps_per_m=10.669 support_feet=3.155 taps_per_m=5.382 swing_apex=0.101 step_rate_hz=2.458 gait_hz=0.394 clock_hz=0.314 cadence_ratio=1.253 hz_speed_slope=0.223 offset_R=0.086
       back: steps_per_m=12.484 support_feet=3.117 taps_per_m=9.614 swing_apex=0.105 step_rate_hz=2.750 gait_hz=0.421 clock_hz=0.315 cadence_ratio=1.335 hz_speed_slope=0.142 offset_R=0.095
       diag: steps_per_m=11.573 support_feet=3.066 taps_per_m=7.107 swing_apex=0.094 step_rate_hz=2.917 gait_hz=0.446 clock_hz=0.357 cadence_ratio=1.248 hz_speed_slope=-0.309 offset_R=0.088
       turn: steps_per_rad=19.121 support_feet=3.218 taps_per_rad=8.906 swing_apex=0.140 step_rate_hz=2.500 gait_hz=0.415 clock_hz=0.353 cadence_ratio=1.182 steps_per_rad_pre=13.507 steps_per_rad_post=5.716 offset_R=0.023
  move_turn: steps_per_rad=31.934 support_feet=3.089 taps_per_rad=27.362 swing_apex=0.103 step_rate_hz=2.667 gait_hz=0.363 clock_hz=0.500 cadence_ratio=0.726 hz_speed_slope=-0.050 offset_R=0.041

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
