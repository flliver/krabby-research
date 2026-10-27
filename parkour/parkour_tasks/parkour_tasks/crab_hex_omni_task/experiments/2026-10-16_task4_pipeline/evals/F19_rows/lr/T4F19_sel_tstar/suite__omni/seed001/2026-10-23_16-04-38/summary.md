```
=== crab-hex gait eval ===
scenario   : suite__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7208772604003482  p25=0.6838535751257493  p75=0.772810213577571
tippy_tap_fraction median=0.4578313253012048
slip_ratio         median=0.24090620840967175
tracking_ratio     median=1.005754481623621  (achieved/commanded vx, walking holds; n=700)
schedule_completion_rate=1.0
terminations={'schedule_complete': 1000}
facing_error_rms   median=0.047840103868582534  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7686066366951505 (n=981)
     stand: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7867098310677014 (n=100)
       fwd: cmd 0.25 -> achieved 0.253 m/s | tripod median=0.6591755437807882 (n=100)
     lat_l: cmd 0.22 -> achieved 0.235 m/s | tripod median=0.6760898770856918 (n=100)
     lat_r: cmd 0.22 -> achieved 0.232 m/s | tripod median=0.7645119824818494 (n=100)
      back: cmd 0.22 -> achieved 0.217 m/s | tripod median=0.7058621846746773 (n=100)
      diag: cmd 0.25 -> achieved 0.253 m/s | tripod median=0.6953284042226091 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.004 m/s | tripod median=0.8000145350862803 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.007 m/s | tripod median=0.7931971583487079 (n=100)
  move_turn: cmd 0.22 -> achieved 0.158 m/s | tripod median=0.6702427862259357 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.166 m/s | tripod median=0.7092658217297907 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.013 initial=0.008 final=0.012 max=0.028 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.001 yaw_rate_ratio=n/a
     stand: err rms=0.024 initial=0.010 final=0.023 max=0.055 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a
       fwd: err rms=0.044 initial=0.011 final=0.035 max=0.091 closed=n/a | settle=0.000 (n=100/100) | progress=-0.001 net=0.002 yaw_rate_ratio=0.394
     lat_l: err rms=0.018 initial=0.012 final=0.012 max=0.042 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
     lat_r: err rms=0.032 initial=0.012 final=0.032 max=0.068 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a
      back: err rms=0.056 initial=0.012 final=0.084 max=0.116 closed=n/a | settle=0.000 (n=100/100) | progress=-0.002 net=-0.002 yaw_rate_ratio=0.355
      diag: err rms=0.031 initial=0.010 final=0.033 max=0.075 closed=n/a | settle=0.000 (n=99/100) | progress=-0.002 net=0.001 yaw_rate_ratio=0.379
  turn_l90: err rms=0.783 initial=1.572 final=0.022 max=1.574 closed=0.986 | settle=8.140 (n=100/100) | progress=0.175 net=0.132 yaw_rate_ratio=0.661
  turn_r90: err rms=0.800 initial=1.573 final=0.021 max=1.575 closed=0.986 | settle=8.380 (n=100/100) | progress=0.171 net=0.130 yaw_rate_ratio=0.644
  move_turn: err rms=1.088 initial=1.568 final=0.592 max=1.571 closed=0.619 | settle=n/a (n=0/100) | progress=0.080 net=0.080 yaw_rate_ratio=0.422
  move_turn_r: err rms=1.030 initial=1.571 final=0.381 max=1.577 closed=0.757 | settle=11.920 (n=2/100) | progress=0.100 net=0.100 yaw_rate_ratio=0.450

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.003 tripod=0.787 | closed=n/a progress=n/a net=0.000 | drift=0.034 dir_err=n/a
        fwd: n=100 surv=1.000 (fwd=1.00) | ratio=1.012 achieved=0.253 tripod=0.659 | closed=n/a progress=-0.001 net=0.002 | drift=0.252 dir_err=0.011
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=1.061 achieved=0.233 tripod=0.720 | closed=n/a progress=n/a net=0.001 | drift=0.234 dir_err=0.020
       back: n=100 surv=1.000 (back=1.00) | ratio=0.988 achieved=0.217 tripod=0.706 | closed=n/a progress=-0.002 net=-0.002 | drift=0.219 dir_err=0.019
       diag: n=100 surv=1.000 (diag=1.00) | ratio=1.014 achieved=0.253 tripod=0.695 | closed=n/a progress=-0.002 net=0.001 | drift=0.257 dir_err=0.061
       turn: n=200 surv=1.000 (turn_l90=1.00 turn_r90=1.00) | ratio=n/a achieved=0.001 tripod=0.798 | closed=0.986 progress=0.173 net=0.131 | drift=0.046 dir_err=n/a
  move_turn: n=200 surv=1.000 (move_turn=1.00 move_turn_r=1.00) | ratio=0.740 achieved=0.163 tripod=0.690 | closed=0.685 progress=0.089 net=0.090 | drift=0.176 dir_err=0.161

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.208 action_fft_freq=1.997 joint_speed_mean=0.366 rod_speed_mean=0.029 contact_switch_hz=1.792 foot_slip=0.073 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.013 settle_six_down_s=12.000 foot_disp_end=0.084 work_per_s=287.699
        fwd: action_jitter=0.133 action_fft_freq=1.158 joint_speed_mean=0.753 rod_speed_mean=0.029 contact_switch_hz=1.951 foot_slip=0.095 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=23.260 slip_per_m=1.106 work_per_m=1086.449 cam_work_per_m=400.650
       side: action_jitter=0.108 action_fft_freq=1.226 joint_speed_mean=0.390 rod_speed_mean=0.029 contact_switch_hz=1.472 foot_slip=0.073 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=19.081 slip_per_m=0.972 work_per_m=1013.537 cam_work_per_m=199.120
       back: action_jitter=0.142 action_fft_freq=1.256 joint_speed_mean=0.706 rod_speed_mean=0.029 contact_switch_hz=1.903 foot_slip=0.093 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=26.753 slip_per_m=1.289 work_per_m=1193.200 cam_work_per_m=392.713
       diag: action_jitter=0.122 action_fft_freq=1.268 joint_speed_mean=0.581 rod_speed_mean=0.029 contact_switch_hz=1.750 foot_slip=0.088 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=20.898 slip_per_m=1.035 work_per_m=1003.102 cam_work_per_m=216.502
       turn: action_jitter=0.152 action_fft_freq=1.274 joint_speed_mean=0.681 rod_speed_mean=0.029 contact_switch_hz=1.667 foot_slip=0.074 wz_active_frac=0.695 clock_restarts=0.000 swings_per_rad=38.003 slip_per_rad=1.759 work_per_rad=1841.912 cam_work_per_rad=592.623
  move_turn: action_jitter=0.121 action_fft_freq=1.126 joint_speed_mean=0.685 rod_speed_mean=0.028 contact_switch_hz=1.931 foot_slip=0.101 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=64.099 slip_per_rad=3.344 work_per_rad=3222.595 cam_work_per_rad=1117.789

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.152 steps_per_s=2.500 gait_hz=0.427 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=10.648 support_feet=2.943 taps_per_m=12.492 swing_apex=0.120 step_rate_hz=2.667 gait_hz=0.397 clock_hz=0.357 cadence_ratio=1.111 hz_speed_slope=-0.025 offset_R=0.060
       side: steps_per_m=11.090 support_feet=3.063 taps_per_m=7.853 swing_apex=0.099 step_rate_hz=2.583 gait_hz=0.418 clock_hz=0.314 cadence_ratio=1.330 hz_speed_slope=5.266 offset_R=0.036
       back: steps_per_m=12.870 support_feet=2.963 taps_per_m=13.667 swing_apex=0.110 step_rate_hz=2.750 gait_hz=0.421 clock_hz=0.314 cadence_ratio=1.340 hz_speed_slope=-0.220 offset_R=0.083
       diag: steps_per_m=11.857 support_feet=2.946 taps_per_m=8.877 swing_apex=0.094 step_rate_hz=3.000 gait_hz=0.470 clock_hz=0.357 cadence_ratio=1.316 hz_speed_slope=-0.664 offset_R=0.086
       turn: steps_per_rad=19.571 support_feet=3.108 taps_per_rad=18.118 swing_apex=0.138 step_rate_hz=2.583 gait_hz=0.433 clock_hz=0.346 cadence_ratio=1.254 steps_per_rad_pre=13.556 steps_per_rad_post=5.780 offset_R=0.011
  move_turn: steps_per_rad=30.586 support_feet=3.002 taps_per_rad=33.374 swing_apex=0.110 step_rate_hz=2.708 gait_hz=0.372 clock_hz=0.500 cadence_ratio=0.745 hz_speed_slope=-0.049 offset_R=0.027

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
