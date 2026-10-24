```
=== crab-hex gait eval ===
scenario   : suite__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_28000.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6349539386948873  p25=0.5626821793070107  p75=0.6876354199548314
tippy_tap_fraction median=0.3611111111111111
slip_ratio         median=0.19436081077391137
tracking_ratio     median=1.0098107902379796  (achieved/commanded vx, walking holds; n=700)
schedule_completion_rate=0.98
terminations={'schedule_complete': 980, 'fall': 20}
facing_error_rms   median=0.05883575583310385  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.000 m/s | tripod median=0.27962669462501655 (n=781)
     stand: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7171802313593378 (n=100)
       fwd: cmd 0.25 -> achieved 0.256 m/s | tripod median=0.5789476145286976 (n=100)
     lat_l: cmd 0.22 -> achieved 0.223 m/s | tripod median=0.6404973091803325 (n=100)
     lat_r: cmd 0.22 -> achieved 0.223 m/s | tripod median=0.7243137944193081 (n=100)
      back: cmd 0.22 -> achieved 0.225 m/s | tripod median=0.6549015002110419 (n=100)
      diag: cmd 0.25 -> achieved 0.259 m/s | tripod median=0.6470185900654649 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.012 m/s | tripod median=0.7337566210833987 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.009 m/s | tripod median=0.7363947876406594 (n=100)
  move_turn: cmd 0.22 -> achieved 0.184 m/s | tripod median=0.42574888930883564 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.170 m/s | tripod median=0.5619641863871179 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.011 initial=0.009 final=0.013 max=0.025 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.001 yaw_rate_ratio=n/a
     stand: err rms=0.029 initial=0.014 final=0.037 max=0.057 closed=n/a | settle=0.000 (n=100/100) | progress=-0.047 net=-0.001 yaw_rate_ratio=0.411
       fwd: err rms=0.049 initial=0.014 final=0.067 max=0.115 closed=n/a | settle=0.000 (n=97/100) | progress=-0.005 net=0.003 yaw_rate_ratio=0.769
     lat_l: err rms=0.019 initial=0.012 final=0.018 max=0.051 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.002 yaw_rate_ratio=n/a
     lat_r: err rms=0.027 initial=0.014 final=0.020 max=0.062 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
      back: err rms=0.079 initial=0.013 final=0.097 max=0.156 closed=n/a | settle=6.760 (n=96/100) | progress=-0.001 net=-0.004 yaw_rate_ratio=0.720
      diag: err rms=0.035 initial=0.013 final=0.031 max=0.086 closed=n/a | settle=0.000 (n=100/100) | progress=-0.003 net=0.002 yaw_rate_ratio=0.464
  turn_l90: err rms=0.798 initial=1.561 final=0.016 max=1.568 closed=0.989 | settle=8.600 (n=100/100) | progress=0.166 net=0.130 yaw_rate_ratio=0.638
  turn_r90: err rms=0.816 initial=1.579 final=0.027 max=1.585 closed=0.983 | settle=8.720 (n=99/100) | progress=0.165 net=0.129 yaw_rate_ratio=0.640
  move_turn: err rms=1.207 initial=1.562 final=0.959 max=1.578 closed=0.386 | settle=n/a (n=0/100) | progress=0.052 net=0.053 yaw_rate_ratio=0.458
  move_turn_r: err rms=1.023 initial=1.581 final=0.430 max=1.585 closed=0.726 | settle=n/a (n=0/100) | progress=0.095 net=0.096 yaw_rate_ratio=0.466

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=0.980 (stand=0.98) | ratio=n/a achieved=0.003 tripod=0.717 | closed=n/a progress=-0.047 net=-0.001 | drift=0.033 dir_err=n/a
        fwd: n=100 surv=1.000 (fwd=1.00) | ratio=1.025 achieved=0.256 tripod=0.579 | closed=n/a progress=-0.005 net=0.003 | drift=0.257 dir_err=0.020
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=1.012 achieved=0.223 tripod=0.680 | closed=n/a progress=n/a net=0.001 | drift=0.225 dir_err=0.016
       back: n=100 surv=0.990 (back=0.99) | ratio=1.022 achieved=0.225 tripod=0.655 | closed=n/a progress=-0.001 net=-0.004 | drift=0.226 dir_err=0.015
       diag: n=100 surv=0.990 (diag=0.99) | ratio=1.036 achieved=0.259 tripod=0.647 | closed=n/a progress=-0.003 net=0.002 | drift=0.261 dir_err=0.032
       turn: n=200 surv=1.000 (turn_l90=1.00 turn_r90=1.00) | ratio=n/a achieved=-0.003 tripod=0.735 | closed=0.986 progress=0.165 net=0.130 | drift=0.049 dir_err=n/a
  move_turn: n=200 surv=0.920 (move_turn=0.87 move_turn_r=0.97) | ratio=0.805 achieved=0.177 tripod=0.501 | closed=0.492 progress=0.069 net=0.070 | drift=0.192 dir_err=0.114

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.108 action_fft_freq=1.238 joint_speed_mean=0.362 rod_speed_mean=0.028 contact_switch_hz=1.139 foot_slip=0.091 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.042 settle_six_down_s=12.000 foot_disp_end=0.110 work_per_s=432.700
        fwd: action_jitter=0.081 action_fft_freq=0.909 joint_speed_mean=0.754 rod_speed_mean=0.029 contact_switch_hz=1.576 foot_slip=0.115 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=19.134 slip_per_m=1.471 work_per_m=1584.819 cam_work_per_m=684.852
       side: action_jitter=0.067 action_fft_freq=0.962 joint_speed_mean=0.384 rod_speed_mean=0.028 contact_switch_hz=1.243 foot_slip=0.101 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=16.791 slip_per_m=1.491 work_per_m=1646.993 cam_work_per_m=316.353
       back: action_jitter=0.088 action_fft_freq=0.936 joint_speed_mean=0.718 rod_speed_mean=0.029 contact_switch_hz=1.514 foot_slip=0.114 wz_active_frac=0.028 clock_restarts=0.000 swings_per_m=20.606 slip_per_m=1.657 work_per_m=1692.991 cam_work_per_m=627.453
       diag: action_jitter=0.083 action_fft_freq=1.006 joint_speed_mean=0.577 rod_speed_mean=0.029 contact_switch_hz=1.583 foot_slip=0.120 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=18.621 slip_per_m=1.493 work_per_m=1433.968 cam_work_per_m=319.365
       turn: action_jitter=0.087 action_fft_freq=0.928 joint_speed_mean=0.692 rod_speed_mean=0.029 contact_switch_hz=1.208 foot_slip=0.095 wz_active_frac=0.727 clock_restarts=0.000 swings_per_rad=28.427 slip_per_rad=2.438 work_per_rad=2661.524 cam_work_per_rad=885.708
  move_turn: action_jitter=0.078 action_fft_freq=0.927 joint_speed_mean=0.670 rod_speed_mean=0.028 contact_switch_hz=1.625 foot_slip=0.127 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=67.756 slip_per_rad=5.919 work_per_rad=5955.955 cam_work_per_rad=2989.635

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.428 steps_per_s=2.167 gait_hz=0.360 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=10.981 support_feet=3.232 taps_per_m=7.992 swing_apex=0.111 step_rate_hz=2.750 gait_hz=0.385 clock_hz=0.357 cadence_ratio=1.077 hz_speed_slope=-0.411 offset_R=0.156
       side: steps_per_m=11.430 support_feet=3.278 taps_per_m=6.002 swing_apex=0.092 step_rate_hz=2.500 gait_hz=0.375 clock_hz=0.314 cadence_ratio=1.194 hz_speed_slope=0.757 offset_R=0.133
       back: steps_per_m=12.300 support_feet=3.255 taps_per_m=8.307 swing_apex=0.105 step_rate_hz=2.750 gait_hz=0.403 clock_hz=0.316 cadence_ratio=1.262 hz_speed_slope=0.180 offset_R=0.121
       diag: steps_per_m=11.565 support_feet=3.184 taps_per_m=6.998 swing_apex=0.091 step_rate_hz=3.000 gait_hz=0.439 clock_hz=0.357 cadence_ratio=1.230 hz_speed_slope=-0.900 offset_R=0.086
       turn: steps_per_rad=19.834 support_feet=3.314 taps_per_rad=8.740 swing_apex=0.136 step_rate_hz=2.583 gait_hz=0.403 clock_hz=0.361 cadence_ratio=1.123 steps_per_rad_pre=15.003 steps_per_rad_post=4.615 offset_R=0.087
  move_turn: steps_per_rad=38.849 support_feet=3.183 taps_per_rad=28.631 swing_apex=0.092 step_rate_hz=2.750 gait_hz=0.360 clock_hz=0.500 cadence_ratio=0.720 hz_speed_slope=0.020 offset_R=0.112

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
