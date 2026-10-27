```
=== crab-hex gait eval ===
scenario   : suite__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-21_23-06-12_seed19_X06_p2a_A15pB/model_19994.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7481037940423583  p25=0.6973075674559428  p75=0.7937129867807908
tippy_tap_fraction median=0.4444444444444444
slip_ratio         median=0.23387161667868842
tracking_ratio     median=1.0328734905090031  (achieved/commanded vx, walking holds; n=700)
schedule_completion_rate=1.0
terminations={'schedule_complete': 1000}
facing_error_rms   median=0.0425086223426011  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7710998153737153 (n=977)
     stand: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.797374791651819 (n=100)
       fwd: cmd 0.25 -> achieved 0.262 m/s | tripod median=0.6039734435509971 (n=100)
     lat_l: cmd 0.22 -> achieved 0.234 m/s | tripod median=0.7260896122301725 (n=100)
     lat_r: cmd 0.22 -> achieved 0.235 m/s | tripod median=0.7928313365008797 (n=100)
      back: cmd 0.22 -> achieved 0.233 m/s | tripod median=0.7431594157459324 (n=100)
      diag: cmd 0.25 -> achieved 0.249 m/s | tripod median=0.7445239800580453 (n=100)
  turn_l90: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.8098413683891217 (n=100)
  turn_r90: cmd 0.00 -> achieved -0.002 m/s | tripod median=0.8209050016963297 (n=100)
  move_turn: cmd 0.22 -> achieved 0.168 m/s | tripod median=0.6209809837019384 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.177 m/s | tripod median=0.7069259377653762 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.018 initial=0.006 final=0.014 max=0.034 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.001 yaw_rate_ratio=n/a
     stand: err rms=0.024 initial=0.014 final=0.021 max=0.055 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a
       fwd: err rms=0.038 initial=0.014 final=0.039 max=0.087 closed=n/a | settle=0.000 (n=100/100) | progress=-0.002 net=0.000 yaw_rate_ratio=0.310
     lat_l: err rms=0.025 initial=0.011 final=0.028 max=0.050 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a
     lat_r: err rms=0.023 initial=0.014 final=0.019 max=0.047 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
      back: err rms=0.036 initial=0.017 final=0.036 max=0.080 closed=n/a | settle=0.000 (n=100/100) | progress=-0.001 net=0.004 yaw_rate_ratio=0.201
      diag: err rms=0.037 initial=0.014 final=0.035 max=0.081 closed=n/a | settle=0.000 (n=100/100) | progress=-0.001 net=0.000 yaw_rate_ratio=0.370
  turn_l90: err rms=0.819 initial=1.570 final=0.091 max=1.571 closed=0.942 | settle=10.080 (n=100/100) | progress=0.149 net=0.123 yaw_rate_ratio=0.588
  turn_r90: err rms=0.830 initial=1.581 final=0.103 max=1.583 closed=0.934 | settle=10.280 (n=96/100) | progress=0.143 net=0.123 yaw_rate_ratio=0.588
  move_turn: err rms=1.128 initial=1.570 final=0.637 max=1.571 closed=0.594 | settle=n/a (n=0/100) | progress=0.077 net=0.077 yaw_rate_ratio=0.357
  move_turn_r: err rms=1.114 initial=1.576 final=0.645 max=1.582 closed=0.590 | settle=n/a (n=0/100) | progress=0.077 net=0.078 yaw_rate_ratio=0.387

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.002 tripod=0.797 | closed=n/a progress=n/a net=0.000 | drift=0.038 dir_err=n/a
        fwd: n=100 surv=1.000 (fwd=1.00) | ratio=1.046 achieved=0.262 tripod=0.604 | closed=n/a progress=-0.002 net=0.000 | drift=0.259 dir_err=0.063
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=1.065 achieved=0.234 tripod=0.766 | closed=n/a progress=n/a net=0.001 | drift=0.233 dir_err=0.012
       back: n=100 surv=1.000 (back=1.00) | ratio=1.061 achieved=0.233 tripod=0.743 | closed=n/a progress=-0.001 net=0.004 | drift=0.233 dir_err=0.046
       diag: n=100 surv=1.000 (diag=1.00) | ratio=0.997 achieved=0.249 tripod=0.745 | closed=n/a progress=-0.001 net=0.000 | drift=0.251 dir_err=0.035
       turn: n=200 surv=1.000 (turn_l90=1.00 turn_r90=1.00) | ratio=n/a achieved=-0.000 tripod=0.815 | closed=0.938 progress=0.146 net=0.123 | drift=0.049 dir_err=n/a
  move_turn: n=200 surv=1.000 (move_turn=1.00 move_turn_r=1.00) | ratio=0.786 achieved=0.173 tripod=0.669 | closed=0.592 progress=0.077 net=0.078 | drift=0.179 dir_err=0.092

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.203 action_fft_freq=1.898 joint_speed_mean=0.337 rod_speed_mean=0.030 contact_switch_hz=1.583 foot_slip=0.068 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.003 settle_six_down_s=12.000 foot_disp_end=0.081 work_per_s=238.767
        fwd: action_jitter=0.155 action_fft_freq=1.234 joint_speed_mean=0.760 rod_speed_mean=0.029 contact_switch_hz=1.993 foot_slip=0.105 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=23.128 slip_per_m=1.193 work_per_m=1127.027 cam_work_per_m=402.461
       side: action_jitter=0.136 action_fft_freq=1.426 joint_speed_mean=0.345 rod_speed_mean=0.029 contact_switch_hz=1.569 foot_slip=0.073 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=20.423 slip_per_m=0.975 work_per_m=1017.528 cam_work_per_m=200.156
       back: action_jitter=0.167 action_fft_freq=1.281 joint_speed_mean=0.720 rod_speed_mean=0.029 contact_switch_hz=1.861 foot_slip=0.100 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=23.979 slip_per_m=1.282 work_per_m=1139.225 cam_work_per_m=395.433
       diag: action_jitter=0.156 action_fft_freq=1.344 joint_speed_mean=0.611 rod_speed_mean=0.029 contact_switch_hz=1.847 foot_slip=0.084 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=22.381 slip_per_m=1.035 work_per_m=992.140 cam_work_per_m=250.314
       turn: action_jitter=0.152 action_fft_freq=1.243 joint_speed_mean=0.696 rod_speed_mean=0.029 contact_switch_hz=1.472 foot_slip=0.092 wz_active_frac=0.828 clock_restarts=1.000 swings_per_rad=35.950 slip_per_rad=2.324 work_per_rad=2169.405 cam_work_per_rad=654.546
  move_turn: action_jitter=0.152 action_fft_freq=1.279 joint_speed_mean=0.637 rod_speed_mean=0.028 contact_switch_hz=1.972 foot_slip=0.109 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=76.286 slip_per_rad=4.185 work_per_rad=4108.625 cam_work_per_rad=1172.672

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.057 steps_per_s=2.833 gait_hz=0.482 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=9.989 support_feet=2.937 taps_per_m=12.749 swing_apex=0.124 step_rate_hz=2.583 gait_hz=0.372 clock_hz=0.357 cadence_ratio=1.042 hz_speed_slope=0.274 offset_R=0.075
       side: steps_per_m=12.824 support_feet=3.086 taps_per_m=7.461 swing_apex=0.085 step_rate_hz=3.000 gait_hz=0.494 clock_hz=0.314 cadence_ratio=1.573 hz_speed_slope=1.913 offset_R=0.030
       back: steps_per_m=10.915 support_feet=2.970 taps_per_m=12.879 swing_apex=0.123 step_rate_hz=2.500 gait_hz=0.403 clock_hz=0.314 cadence_ratio=1.280 hz_speed_slope=-0.604 offset_R=0.082
       diag: steps_per_m=12.194 support_feet=3.043 taps_per_m=9.952 swing_apex=0.098 step_rate_hz=3.000 gait_hz=0.494 clock_hz=0.357 cadence_ratio=1.384 hz_speed_slope=-0.438 offset_R=0.058
       turn: steps_per_rad=20.876 support_feet=3.102 taps_per_rad=14.668 swing_apex=0.129 step_rate_hz=2.583 gait_hz=0.433 clock_hz=0.407 cadence_ratio=1.061 steps_per_rad_pre=16.705 steps_per_rad_post=4.092 offset_R=0.063
  move_turn: steps_per_rad=34.448 support_feet=2.995 taps_per_rad=41.889 swing_apex=0.126 step_rate_hz=2.667 gait_hz=0.372 clock_hz=0.500 cadence_ratio=0.745 hz_speed_slope=-0.697 offset_R=0.076

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
