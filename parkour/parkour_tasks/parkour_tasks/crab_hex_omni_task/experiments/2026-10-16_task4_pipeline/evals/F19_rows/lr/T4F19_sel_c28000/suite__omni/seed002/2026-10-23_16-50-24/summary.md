```
=== crab-hex gait eval ===
scenario   : suite__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_28000.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7099714330229325  p25=0.6710778779728807  p75=0.7553312624324076
tippy_tap_fraction median=0.42340286831812257
slip_ratio         median=0.23593340476589011
tracking_ratio     median=0.9938378616396675  (achieved/commanded vx, walking holds; n=700)
schedule_completion_rate=1.0
terminations={'schedule_complete': 1000}
facing_error_rms   median=0.06158351699227031  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.6703349177208406 (n=937)
     stand: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7614363939399127 (n=100)
       fwd: cmd 0.25 -> achieved 0.247 m/s | tripod median=0.6798185981393693 (n=100)
     lat_l: cmd 0.22 -> achieved 0.230 m/s | tripod median=0.6603639356491221 (n=100)
     lat_r: cmd 0.22 -> achieved 0.231 m/s | tripod median=0.762155555489705 (n=100)
      back: cmd 0.22 -> achieved 0.216 m/s | tripod median=0.7045198903827454 (n=100)
      diag: cmd 0.25 -> achieved 0.253 m/s | tripod median=0.6921723842566461 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.006 m/s | tripod median=0.7979649378636327 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.007 m/s | tripod median=0.7864018587507671 (n=100)
  move_turn: cmd 0.22 -> achieved 0.162 m/s | tripod median=0.6526352886766779 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.164 m/s | tripod median=0.7240356769205908 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.014 initial=0.009 final=0.017 max=0.030 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=-0.000 yaw_rate_ratio=n/a
     stand: err rms=0.031 initial=0.016 final=0.037 max=0.064 closed=n/a | settle=0.000 (n=100/100) | progress=-0.003 net=-0.000 yaw_rate_ratio=0.497
       fwd: err rms=0.044 initial=0.017 final=0.038 max=0.111 closed=n/a | settle=0.000 (n=99/100) | progress=-0.002 net=0.003 yaw_rate_ratio=0.685
     lat_l: err rms=0.021 initial=0.016 final=0.020 max=0.049 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.002 yaw_rate_ratio=n/a
     lat_r: err rms=0.035 initial=0.019 final=0.034 max=0.072 closed=n/a | settle=0.000 (n=100/100) | progress=-0.004 net=0.001 yaw_rate_ratio=0.134
      back: err rms=0.088 initial=0.015 final=0.100 max=0.156 closed=n/a | settle=5.970 (n=100/100) | progress=-0.003 net=-0.005 yaw_rate_ratio=0.562
      diag: err rms=0.039 initial=0.014 final=0.056 max=0.090 closed=n/a | settle=0.000 (n=100/100) | progress=-0.002 net=0.004 yaw_rate_ratio=0.347
  turn_l90: err rms=0.781 initial=1.566 final=0.019 max=1.570 closed=0.988 | settle=8.220 (n=100/100) | progress=0.172 net=0.130 yaw_rate_ratio=0.666
  turn_r90: err rms=0.817 initial=1.584 final=0.022 max=1.587 closed=0.986 | settle=8.520 (n=100/100) | progress=0.169 net=0.130 yaw_rate_ratio=0.644
  move_turn: err rms=1.093 initial=1.560 final=0.635 max=1.566 closed=0.595 | settle=n/a (n=0/100) | progress=0.077 net=0.078 yaw_rate_ratio=0.410
  move_turn_r: err rms=1.009 initial=1.581 final=0.320 max=1.583 closed=0.801 | settle=11.820 (n=3/100) | progress=0.106 net=0.106 yaw_rate_ratio=0.463

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.003 tripod=0.761 | closed=n/a progress=-0.003 net=-0.000 | drift=0.035 dir_err=n/a
        fwd: n=100 surv=1.000 (fwd=1.00) | ratio=0.987 achieved=0.247 tripod=0.680 | closed=n/a progress=-0.002 net=0.003 | drift=0.247 dir_err=0.018
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=1.047 achieved=0.230 tripod=0.714 | closed=n/a progress=-0.004 net=0.001 | drift=0.232 dir_err=0.018
       back: n=100 surv=1.000 (back=1.00) | ratio=0.983 achieved=0.216 tripod=0.705 | closed=n/a progress=-0.003 net=-0.005 | drift=0.217 dir_err=0.031
       diag: n=100 surv=1.000 (diag=1.00) | ratio=1.012 achieved=0.253 tripod=0.692 | closed=n/a progress=-0.002 net=0.004 | drift=0.258 dir_err=0.058
       turn: n=200 surv=1.000 (turn_l90=1.00 turn_r90=1.00) | ratio=n/a achieved=0.000 tripod=0.792 | closed=0.987 progress=0.170 net=0.130 | drift=0.048 dir_err=n/a
  move_turn: n=200 surv=1.000 (move_turn=1.00 move_turn_r=1.00) | ratio=0.739 achieved=0.162 tripod=0.693 | closed=0.724 progress=0.094 net=0.095 | drift=0.177 dir_err=0.173

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.142 action_fft_freq=1.640 joint_speed_mean=0.360 rod_speed_mean=0.029 contact_switch_hz=1.590 foot_slip=0.076 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.010 settle_six_down_s=12.000 foot_disp_end=0.118 work_per_s=298.982
        fwd: action_jitter=0.105 action_fft_freq=1.062 joint_speed_mean=0.754 rod_speed_mean=0.029 contact_switch_hz=1.826 foot_slip=0.094 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=22.114 slip_per_m=1.143 work_per_m=1115.986 cam_work_per_m=420.954
       side: action_jitter=0.083 action_fft_freq=1.126 joint_speed_mean=0.385 rod_speed_mean=0.029 contact_switch_hz=1.389 foot_slip=0.075 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=18.662 slip_per_m=0.996 work_per_m=1045.338 cam_work_per_m=194.974
       back: action_jitter=0.114 action_fft_freq=1.147 joint_speed_mean=0.703 rod_speed_mean=0.029 contact_switch_hz=1.833 foot_slip=0.094 wz_active_frac=0.047 clock_restarts=0.000 swings_per_m=25.654 slip_per_m=1.285 work_per_m=1200.515 cam_work_per_m=400.975
       diag: action_jitter=0.097 action_fft_freq=1.192 joint_speed_mean=0.567 rod_speed_mean=0.029 contact_switch_hz=1.639 foot_slip=0.087 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=19.434 slip_per_m=1.023 work_per_m=1015.640 cam_work_per_m=205.806
       turn: action_jitter=0.112 action_fft_freq=1.119 joint_speed_mean=0.683 rod_speed_mean=0.029 contact_switch_hz=1.556 foot_slip=0.077 wz_active_frac=0.703 clock_restarts=0.000 swings_per_rad=35.878 slip_per_rad=1.854 work_per_rad=1882.461 cam_work_per_rad=622.178
  move_turn: action_jitter=0.093 action_fft_freq=1.034 joint_speed_mean=0.684 rod_speed_mean=0.028 contact_switch_hz=1.833 foot_slip=0.102 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=58.716 slip_per_rad=3.246 work_per_rad=3111.623 cam_work_per_rad=1099.584

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.207 steps_per_s=2.333 gait_hz=0.385 clock_hz=0.000 cadence_ratio=6.269
        fwd: steps_per_m=10.740 support_feet=2.975 taps_per_m=11.382 swing_apex=0.121 step_rate_hz=2.667 gait_hz=0.397 clock_hz=0.357 cadence_ratio=1.111 hz_speed_slope=-0.317 offset_R=0.156
       side: steps_per_m=11.149 support_feet=3.047 taps_per_m=7.277 swing_apex=0.102 step_rate_hz=2.542 gait_hz=0.400 clock_hz=0.314 cadence_ratio=1.272 hz_speed_slope=1.443 offset_R=0.010
       back: steps_per_m=12.910 support_feet=2.966 taps_per_m=12.715 swing_apex=0.108 step_rate_hz=2.750 gait_hz=0.421 clock_hz=0.317 cadence_ratio=1.316 hz_speed_slope=-0.317 offset_R=0.109
       diag: steps_per_m=12.140 support_feet=2.951 taps_per_m=7.103 swing_apex=0.094 step_rate_hz=3.083 gait_hz=0.464 clock_hz=0.357 cadence_ratio=1.299 hz_speed_slope=0.088 offset_R=0.061
       turn: steps_per_rad=19.810 support_feet=3.117 taps_per_rad=15.850 swing_apex=0.140 step_rate_hz=2.583 gait_hz=0.439 clock_hz=0.351 cadence_ratio=1.242 steps_per_rad_pre=13.973 steps_per_rad_post=5.778 offset_R=0.086
  move_turn: steps_per_rad=29.438 support_feet=2.998 taps_per_rad=29.108 swing_apex=0.108 step_rate_hz=2.750 gait_hz=0.372 clock_hz=0.500 cadence_ratio=0.745 hz_speed_slope=0.134 offset_R=0.142

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
