```
=== crab-hex gait eval ===
scenario   : suite__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_27000.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6760594797323851  p25=0.6348446421921938  p75=0.7495023899185401
tippy_tap_fraction median=0.359375
slip_ratio         median=0.20194963188989887
tracking_ratio     median=1.0118113915436355  (achieved/commanded vx, walking holds; n=700)
schedule_completion_rate=1.0
terminations={'schedule_complete': 1000}
facing_error_rms   median=0.049122045936457734  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.7176511037281912 (n=984)
     stand: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.7464276853377473 (n=100)
       fwd: cmd 0.25 -> achieved 0.252 m/s | tripod median=0.626021785941854 (n=100)
     lat_l: cmd 0.22 -> achieved 0.231 m/s | tripod median=0.6335407744054113 (n=100)
     lat_r: cmd 0.22 -> achieved 0.232 m/s | tripod median=0.758201056547148 (n=100)
      back: cmd 0.22 -> achieved 0.221 m/s | tripod median=0.6763655295302347 (n=100)
      diag: cmd 0.25 -> achieved 0.256 m/s | tripod median=0.6612429388268951 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.003 m/s | tripod median=0.781188173710376 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.007 m/s | tripod median=0.7656309057326702 (n=100)
  move_turn: cmd 0.22 -> achieved 0.170 m/s | tripod median=0.5625837632252852 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.168 m/s | tripod median=0.6400495729108228 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.012 initial=0.007 final=0.013 max=0.029 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.002 yaw_rate_ratio=n/a
     stand: err rms=0.023 initial=0.012 final=0.026 max=0.055 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.002 yaw_rate_ratio=n/a
       fwd: err rms=0.042 initial=0.013 final=0.037 max=0.101 closed=n/a | settle=0.000 (n=100/100) | progress=-0.007 net=0.003 yaw_rate_ratio=0.914
     lat_l: err rms=0.023 initial=0.013 final=0.023 max=0.057 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.002 yaw_rate_ratio=n/a
     lat_r: err rms=0.023 initial=0.014 final=0.026 max=0.057 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
      back: err rms=0.060 initial=0.013 final=0.085 max=0.130 closed=n/a | settle=0.000 (n=99/100) | progress=0.000 net=-0.003 yaw_rate_ratio=0.645
      diag: err rms=0.034 initial=0.013 final=0.028 max=0.078 closed=n/a | settle=0.000 (n=100/100) | progress=-0.001 net=0.001 yaw_rate_ratio=0.365
  turn_l90: err rms=0.780 initial=1.567 final=0.013 max=1.572 closed=0.991 | settle=8.220 (n=100/100) | progress=0.173 net=0.131 yaw_rate_ratio=0.650
  turn_r90: err rms=0.800 initial=1.573 final=0.020 max=1.579 closed=0.987 | settle=8.420 (n=100/100) | progress=0.170 net=0.131 yaw_rate_ratio=0.649
  move_turn: err rms=1.107 initial=1.565 final=0.677 max=1.571 closed=0.570 | settle=n/a (n=0/100) | progress=0.073 net=0.074 yaw_rate_ratio=0.432
  move_turn_r: err rms=1.016 initial=1.575 final=0.382 max=1.580 closed=0.760 | settle=n/a (n=0/100) | progress=0.100 net=0.100 yaw_rate_ratio=0.461

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.002 tripod=0.746 | closed=n/a progress=n/a net=0.002 | drift=0.030 dir_err=n/a
        fwd: n=100 surv=1.000 (fwd=1.00) | ratio=1.009 achieved=0.252 tripod=0.626 | closed=n/a progress=-0.007 net=0.003 | drift=0.252 dir_err=0.010
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=1.053 achieved=0.232 tripod=0.699 | closed=n/a progress=n/a net=0.001 | drift=0.232 dir_err=0.023
       back: n=100 surv=1.000 (back=1.00) | ratio=1.005 achieved=0.221 tripod=0.676 | closed=n/a progress=0.000 net=-0.003 | drift=0.222 dir_err=0.034
       diag: n=100 surv=1.000 (diag=1.00) | ratio=1.023 achieved=0.256 tripod=0.661 | closed=n/a progress=-0.001 net=0.001 | drift=0.258 dir_err=0.055
       turn: n=200 surv=1.000 (turn_l90=1.00 turn_r90=1.00) | ratio=n/a achieved=0.002 tripod=0.773 | closed=0.990 progress=0.172 net=0.131 | drift=0.044 dir_err=n/a
  move_turn: n=200 surv=1.000 (move_turn=1.00 move_turn_r=1.00) | ratio=0.767 achieved=0.169 tripod=0.607 | closed=0.652 progress=0.085 net=0.085 | drift=0.183 dir_err=0.113

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.124 action_fft_freq=1.411 joint_speed_mean=0.360 rod_speed_mean=0.028 contact_switch_hz=1.278 foot_slip=0.078 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.027 settle_six_down_s=12.000 foot_disp_end=0.092 work_per_s=353.529
        fwd: action_jitter=0.089 action_fft_freq=0.951 joint_speed_mean=0.754 rod_speed_mean=0.029 contact_switch_hz=1.681 foot_slip=0.109 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=19.877 slip_per_m=1.345 work_per_m=1361.521 cam_work_per_m=529.615
       side: action_jitter=0.073 action_fft_freq=1.014 joint_speed_mean=0.383 rod_speed_mean=0.028 contact_switch_hz=1.208 foot_slip=0.085 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=15.976 slip_per_m=1.159 work_per_m=1261.193 cam_work_per_m=225.755
       back: action_jitter=0.098 action_fft_freq=1.009 joint_speed_mean=0.706 rod_speed_mean=0.029 contact_switch_hz=1.660 foot_slip=0.102 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=22.648 slip_per_m=1.465 work_per_m=1465.138 cam_work_per_m=511.315
       diag: action_jitter=0.086 action_fft_freq=1.080 joint_speed_mean=0.577 rod_speed_mean=0.029 contact_switch_hz=1.583 foot_slip=0.098 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=18.577 slip_per_m=1.178 work_per_m=1212.857 cam_work_per_m=244.600
       turn: action_jitter=0.099 action_fft_freq=1.024 joint_speed_mean=0.675 rod_speed_mean=0.029 contact_switch_hz=1.292 foot_slip=0.079 wz_active_frac=0.700 clock_restarts=0.000 swings_per_rad=29.604 slip_per_rad=1.955 work_per_rad=2178.734 cam_work_per_rad=732.590
  move_turn: action_jitter=0.084 action_fft_freq=0.969 joint_speed_mean=0.672 rod_speed_mean=0.028 contact_switch_hz=1.694 foot_slip=0.111 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=57.844 slip_per_rad=4.102 work_per_rad=4145.651 cam_work_per_rad=1530.869

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.326 steps_per_s=2.500 gait_hz=0.415 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=10.632 support_feet=3.106 taps_per_m=8.993 swing_apex=0.113 step_rate_hz=2.667 gait_hz=0.397 clock_hz=0.357 cadence_ratio=1.111 hz_speed_slope=-0.344 offset_R=0.089
       side: steps_per_m=10.509 support_feet=3.134 taps_per_m=5.476 swing_apex=0.097 step_rate_hz=2.417 gait_hz=0.400 clock_hz=0.314 cadence_ratio=1.272 hz_speed_slope=-0.616 offset_R=0.013
       back: steps_per_m=12.289 support_feet=3.120 taps_per_m=10.567 swing_apex=0.102 step_rate_hz=2.667 gait_hz=0.421 clock_hz=0.314 cadence_ratio=1.340 hz_speed_slope=-0.065 offset_R=0.024
       diag: steps_per_m=11.874 support_feet=3.077 taps_per_m=6.848 swing_apex=0.091 step_rate_hz=3.000 gait_hz=0.464 clock_hz=0.357 cadence_ratio=1.299 hz_speed_slope=-0.443 offset_R=0.133
       turn: steps_per_rad=19.520 support_feet=3.227 taps_per_rad=9.822 swing_apex=0.131 step_rate_hz=2.583 gait_hz=0.439 clock_hz=0.349 cadence_ratio=1.256 steps_per_rad_pre=13.505 steps_per_rad_post=5.754 offset_R=0.043
  move_turn: steps_per_rad=31.744 support_feet=3.098 taps_per_rad=26.566 swing_apex=0.102 step_rate_hz=2.750 gait_hz=0.372 clock_hz=0.500 cadence_ratio=0.745 hz_speed_slope=-0.200 offset_R=0.071

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
