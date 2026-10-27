```
=== crab-hex gait eval ===
scenario   : suite__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_30991.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.682350744160994  p25=0.6355219400210049  p75=0.744812136965751
tippy_tap_fraction median=0.3611111111111111
slip_ratio         median=0.1947179164160811
tracking_ratio     median=1.0315803284526506  (achieved/commanded vx, walking holds; n=700)
schedule_completion_rate=1.0
terminations={'schedule_complete': 1000}
facing_error_rms   median=0.05332059786969006  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.010 m/s | tripod median=0.6419853283078619 (n=957)
     stand: cmd 0.00 -> achieved 0.009 m/s | tripod median=0.7461067116883595 (n=100)
       fwd: cmd 0.25 -> achieved 0.261 m/s | tripod median=0.6424218931322674 (n=100)
     lat_l: cmd 0.22 -> achieved 0.231 m/s | tripod median=0.6502493813895828 (n=100)
     lat_r: cmd 0.22 -> achieved 0.233 m/s | tripod median=0.7647850246324214 (n=100)
      back: cmd 0.22 -> achieved 0.213 m/s | tripod median=0.7048750256285481 (n=100)
      diag: cmd 0.25 -> achieved 0.262 m/s | tripod median=0.6679215067207729 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.005 m/s | tripod median=0.7817025323969727 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.015 m/s | tripod median=0.773465649143761 (n=100)
  move_turn: cmd 0.22 -> achieved 0.175 m/s | tripod median=0.5843027903356224 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.173 m/s | tripod median=0.6392527219243892 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.011 initial=0.005 final=0.013 max=0.026 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.001 yaw_rate_ratio=n/a
     stand: err rms=0.026 initial=0.015 final=0.031 max=0.061 closed=n/a | settle=0.000 (n=100/100) | progress=-0.494 net=0.001 yaw_rate_ratio=2.329
       fwd: err rms=0.037 initial=0.013 final=0.037 max=0.091 closed=n/a | settle=0.000 (n=100/100) | progress=0.007 net=0.003 yaw_rate_ratio=0.925
     lat_l: err rms=0.026 initial=0.015 final=0.032 max=0.056 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.003 yaw_rate_ratio=n/a
     lat_r: err rms=0.027 initial=0.011 final=0.031 max=0.064 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
      back: err rms=0.079 initial=0.013 final=0.085 max=0.147 closed=n/a | settle=0.000 (n=100/100) | progress=-0.001 net=-0.003 yaw_rate_ratio=0.652
      diag: err rms=0.038 initial=0.014 final=0.044 max=0.086 closed=n/a | settle=0.000 (n=100/100) | progress=-0.002 net=0.001 yaw_rate_ratio=0.374
  turn_l90: err rms=0.781 initial=1.564 final=0.017 max=1.571 closed=0.989 | settle=8.260 (n=100/100) | progress=0.172 net=0.131 yaw_rate_ratio=0.660
  turn_r90: err rms=0.808 initial=1.583 final=0.028 max=1.587 closed=0.983 | settle=8.640 (n=100/100) | progress=0.167 net=0.130 yaw_rate_ratio=0.648
  move_turn: err rms=1.151 initial=1.562 final=0.753 max=1.575 closed=0.516 | settle=n/a (n=0/100) | progress=0.066 net=0.067 yaw_rate_ratio=0.421
  move_turn_r: err rms=1.026 initial=1.577 final=0.412 max=1.582 closed=0.738 | settle=n/a (n=0/100) | progress=0.097 net=0.097 yaw_rate_ratio=0.462

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.009 tripod=0.746 | closed=n/a progress=-0.494 net=0.001 | drift=0.031 dir_err=n/a
        fwd: n=100 surv=1.000 (fwd=1.00) | ratio=1.045 achieved=0.261 tripod=0.642 | closed=n/a progress=0.007 net=0.003 | drift=0.261 dir_err=0.008
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=1.055 achieved=0.232 tripod=0.705 | closed=n/a progress=n/a net=0.002 | drift=0.233 dir_err=0.045
       back: n=100 surv=1.000 (back=1.00) | ratio=0.970 achieved=0.213 tripod=0.705 | closed=n/a progress=-0.001 net=-0.003 | drift=0.214 dir_err=0.025
       diag: n=100 surv=1.000 (diag=1.00) | ratio=1.046 achieved=0.262 tripod=0.668 | closed=n/a progress=-0.002 net=0.001 | drift=0.264 dir_err=0.041
       turn: n=200 surv=1.000 (turn_l90=1.00 turn_r90=1.00) | ratio=n/a achieved=0.005 tripod=0.777 | closed=0.986 progress=0.170 net=0.130 | drift=0.047 dir_err=n/a
  move_turn: n=200 surv=1.000 (move_turn=1.00 move_turn_r=1.00) | ratio=0.792 achieved=0.174 tripod=0.609 | closed=0.597 progress=0.077 net=0.078 | drift=0.187 dir_err=0.097

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.124 action_fft_freq=1.410 joint_speed_mean=0.362 rod_speed_mean=0.028 contact_switch_hz=1.222 foot_slip=0.079 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.025 settle_six_down_s=12.000 foot_disp_end=0.158 work_per_s=363.324
        fwd: action_jitter=0.091 action_fft_freq=0.950 joint_speed_mean=0.764 rod_speed_mean=0.029 contact_switch_hz=1.625 foot_slip=0.106 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=18.781 slip_per_m=1.237 work_per_m=1330.401 cam_work_per_m=517.934
       side: action_jitter=0.072 action_fft_freq=0.998 joint_speed_mean=0.386 rod_speed_mean=0.028 contact_switch_hz=1.208 foot_slip=0.083 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=15.743 slip_per_m=1.135 work_per_m=1278.796 cam_work_per_m=229.318
       back: action_jitter=0.101 action_fft_freq=1.032 joint_speed_mean=0.693 rod_speed_mean=0.029 contact_switch_hz=1.611 foot_slip=0.100 wz_active_frac=0.018 clock_restarts=0.000 swings_per_m=22.820 slip_per_m=1.487 work_per_m=1489.518 cam_work_per_m=503.852
       diag: action_jitter=0.086 action_fft_freq=1.063 joint_speed_mean=0.581 rod_speed_mean=0.029 contact_switch_hz=1.590 foot_slip=0.101 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=18.273 slip_per_m=1.190 work_per_m=1190.941 cam_work_per_m=245.082
       turn: action_jitter=0.099 action_fft_freq=1.008 joint_speed_mean=0.686 rod_speed_mean=0.029 contact_switch_hz=1.250 foot_slip=0.082 wz_active_frac=0.708 clock_restarts=0.000 swings_per_rad=28.695 slip_per_rad=2.029 work_per_rad=2262.715 cam_work_per_rad=754.502
  move_turn: action_jitter=0.085 action_fft_freq=0.968 joint_speed_mean=0.675 rod_speed_mean=0.028 contact_switch_hz=1.611 foot_slip=0.111 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=61.327 slip_per_rad=4.294 work_per_rad=4537.832 cam_work_per_rad=1748.289

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.322 steps_per_s=2.333 gait_hz=0.385 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=10.172 support_feet=3.062 taps_per_m=8.647 swing_apex=0.120 step_rate_hz=2.667 gait_hz=0.385 clock_hz=0.357 cadence_ratio=1.077 hz_speed_slope=-0.185 offset_R=0.088
       side: steps_per_m=10.660 support_feet=3.138 taps_per_m=5.080 swing_apex=0.100 step_rate_hz=2.458 gait_hz=0.391 clock_hz=0.314 cadence_ratio=1.243 hz_speed_slope=-0.796 offset_R=0.080
       back: steps_per_m=12.836 support_feet=3.115 taps_per_m=10.073 swing_apex=0.106 step_rate_hz=2.750 gait_hz=0.421 clock_hz=0.315 cadence_ratio=1.321 hz_speed_slope=0.082 offset_R=0.060
       diag: steps_per_m=11.570 support_feet=3.056 taps_per_m=6.685 swing_apex=0.089 step_rate_hz=3.000 gait_hz=0.452 clock_hz=0.357 cadence_ratio=1.265 hz_speed_slope=-0.078 offset_R=0.065
       turn: steps_per_rad=19.136 support_feet=3.215 taps_per_rad=9.613 swing_apex=0.141 step_rate_hz=2.500 gait_hz=0.415 clock_hz=0.352 cadence_ratio=1.183 steps_per_rad_pre=13.642 steps_per_rad_post=5.669 offset_R=0.022
  move_turn: steps_per_rad=31.960 support_feet=3.062 taps_per_rad=27.845 swing_apex=0.104 step_rate_hz=2.583 gait_hz=0.360 clock_hz=0.500 cadence_ratio=0.720 hz_speed_slope=-0.159 offset_R=0.043

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
