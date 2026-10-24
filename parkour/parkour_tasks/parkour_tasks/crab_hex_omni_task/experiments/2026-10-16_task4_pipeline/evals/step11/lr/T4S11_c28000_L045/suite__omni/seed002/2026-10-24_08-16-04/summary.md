```
=== crab-hex gait eval ===
scenario   : suite__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_28000.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7019910912113045  p25=0.6611812638582635  p75=0.7494671498760654
tippy_tap_fraction median=0.38961038961038963
slip_ratio         median=0.2105394157820221
tracking_ratio     median=1.0047828962525611  (achieved/commanded vx, walking holds; n=700)
schedule_completion_rate=1.0
terminations={'schedule_complete': 1000}
facing_error_rms   median=0.06086994878529767  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.645960459337358 (n=950)
     stand: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7531815025035745 (n=100)
       fwd: cmd 0.25 -> achieved 0.251 m/s | tripod median=0.6676995649482085 (n=100)
     lat_l: cmd 0.22 -> achieved 0.230 m/s | tripod median=0.6607732792439901 (n=100)
     lat_r: cmd 0.22 -> achieved 0.232 m/s | tripod median=0.7589144306129338 (n=100)
      back: cmd 0.22 -> achieved 0.219 m/s | tripod median=0.7069879498078602 (n=100)
      diag: cmd 0.25 -> achieved 0.255 m/s | tripod median=0.6890563034379911 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.006 m/s | tripod median=0.7921781124262179 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.007 m/s | tripod median=0.784772240722979 (n=100)
  move_turn: cmd 0.22 -> achieved 0.167 m/s | tripod median=0.6235858870310059 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.166 m/s | tripod median=0.7047460659059054 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.013 initial=0.007 final=0.015 max=0.028 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.000 yaw_rate_ratio=n/a
     stand: err rms=0.032 initial=0.013 final=0.041 max=0.067 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.048 initial=0.015 final=0.043 max=0.110 closed=n/a | settle=0.000 (n=100/100) | progress=-0.000 net=0.004 yaw_rate_ratio=0.665
     lat_l: err rms=0.020 initial=0.016 final=0.019 max=0.047 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.002 yaw_rate_ratio=n/a
     lat_r: err rms=0.030 initial=0.017 final=0.032 max=0.069 closed=n/a | settle=0.000 (n=100/100) | progress=0.006 net=0.000 yaw_rate_ratio=0.395
      back: err rms=0.085 initial=0.016 final=0.095 max=0.149 closed=n/a | settle=0.000 (n=100/100) | progress=-0.002 net=-0.005 yaw_rate_ratio=0.577
      diag: err rms=0.039 initial=0.014 final=0.055 max=0.086 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.005 yaw_rate_ratio=n/a
  turn_l90: err rms=0.783 initial=1.562 final=0.019 max=1.564 closed=0.988 | settle=8.300 (n=100/100) | progress=0.171 net=0.131 yaw_rate_ratio=0.661
  turn_r90: err rms=0.812 initial=1.577 final=0.027 max=1.580 closed=0.983 | settle=8.500 (n=100/100) | progress=0.169 net=0.130 yaw_rate_ratio=0.646
  move_turn: err rms=1.105 initial=1.559 final=0.671 max=1.567 closed=0.570 | settle=n/a (n=0/100) | progress=0.073 net=0.073 yaw_rate_ratio=0.418
  move_turn_r: err rms=1.009 initial=1.577 final=0.323 max=1.585 closed=0.795 | settle=11.820 (n=4/100) | progress=0.105 net=0.105 yaw_rate_ratio=0.462

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.003 tripod=0.753 | closed=n/a progress=n/a net=-0.001 | drift=0.032 dir_err=n/a
        fwd: n=100 surv=1.000 (fwd=1.00) | ratio=1.002 achieved=0.251 tripod=0.668 | closed=n/a progress=-0.000 net=0.004 | drift=0.250 dir_err=0.021
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=1.052 achieved=0.231 tripod=0.703 | closed=n/a progress=0.006 net=0.001 | drift=0.233 dir_err=0.019
       back: n=100 surv=1.000 (back=1.00) | ratio=0.994 achieved=0.219 tripod=0.707 | closed=n/a progress=-0.002 net=-0.005 | drift=0.219 dir_err=0.027
       diag: n=100 surv=1.000 (diag=1.00) | ratio=1.020 achieved=0.255 tripod=0.689 | closed=n/a progress=n/a net=0.005 | drift=0.259 dir_err=0.054
       turn: n=200 surv=1.000 (turn_l90=1.00 turn_r90=1.00) | ratio=n/a achieved=-0.000 tripod=0.789 | closed=0.986 progress=0.170 net=0.131 | drift=0.047 dir_err=n/a
  move_turn: n=200 surv=1.000 (move_turn=1.00 move_turn_r=1.00) | ratio=0.758 achieved=0.167 tripod=0.664 | closed=0.657 progress=0.086 net=0.087 | drift=0.180 dir_err=0.152

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.133 action_fft_freq=1.497 joint_speed_mean=0.359 rod_speed_mean=0.028 contact_switch_hz=1.361 foot_slip=0.077 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.017 settle_six_down_s=12.000 foot_disp_end=0.119 work_per_s=334.024
        fwd: action_jitter=0.099 action_fft_freq=1.007 joint_speed_mean=0.756 rod_speed_mean=0.029 contact_switch_hz=1.708 foot_slip=0.100 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=20.555 slip_per_m=1.215 work_per_m=1246.707 cam_work_per_m=473.018
       side: action_jitter=0.075 action_fft_freq=1.034 joint_speed_mean=0.384 rod_speed_mean=0.028 contact_switch_hz=1.271 foot_slip=0.080 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=16.524 slip_per_m=1.083 work_per_m=1161.969 cam_work_per_m=213.862
       back: action_jitter=0.106 action_fft_freq=1.073 joint_speed_mean=0.703 rod_speed_mean=0.029 contact_switch_hz=1.667 foot_slip=0.097 wz_active_frac=0.030 clock_restarts=0.000 swings_per_m=23.220 slip_per_m=1.363 work_per_m=1337.574 cam_work_per_m=433.767
       diag: action_jitter=0.090 action_fft_freq=1.126 joint_speed_mean=0.563 rod_speed_mean=0.029 contact_switch_hz=1.604 foot_slip=0.093 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=18.888 slip_per_m=1.092 work_per_m=1110.804 cam_work_per_m=222.814
       turn: action_jitter=0.104 action_fft_freq=1.049 joint_speed_mean=0.684 rod_speed_mean=0.029 contact_switch_hz=1.361 foot_slip=0.079 wz_active_frac=0.707 clock_restarts=0.000 swings_per_rad=31.761 slip_per_rad=1.924 work_per_rad=2071.593 cam_work_per_rad=693.914
  move_turn: action_jitter=0.087 action_fft_freq=0.993 joint_speed_mean=0.682 rod_speed_mean=0.028 contact_switch_hz=1.708 foot_slip=0.106 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=58.278 slip_per_rad=3.650 work_per_rad=3617.527 cam_work_per_rad=1328.121

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.274 steps_per_s=2.250 gait_hz=0.378 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=10.448 support_feet=3.024 taps_per_m=9.884 swing_apex=0.120 step_rate_hz=2.583 gait_hz=0.397 clock_hz=0.357 cadence_ratio=1.108 hz_speed_slope=-0.489 offset_R=0.129
       side: steps_per_m=10.904 support_feet=3.111 taps_per_m=5.806 swing_apex=0.103 step_rate_hz=2.500 gait_hz=0.394 clock_hz=0.314 cadence_ratio=1.253 hz_speed_slope=-1.192 offset_R=0.043
       back: steps_per_m=12.673 support_feet=3.043 taps_per_m=10.444 swing_apex=0.108 step_rate_hz=2.750 gait_hz=0.415 clock_hz=0.316 cadence_ratio=1.308 hz_speed_slope=-0.212 offset_R=0.065
       diag: steps_per_m=11.874 support_feet=2.997 taps_per_m=6.746 swing_apex=0.091 step_rate_hz=3.000 gait_hz=0.458 clock_hz=0.357 cadence_ratio=1.282 hz_speed_slope=-0.180 offset_R=0.208
       turn: steps_per_rad=19.540 support_feet=3.173 taps_per_rad=11.943 swing_apex=0.139 step_rate_hz=2.583 gait_hz=0.433 clock_hz=0.352 cadence_ratio=1.230 steps_per_rad_pre=13.810 steps_per_rad_post=5.708 offset_R=0.117
  move_turn: steps_per_rad=29.941 support_feet=3.037 taps_per_rad=28.278 swing_apex=0.110 step_rate_hz=2.667 gait_hz=0.366 clock_hz=0.500 cadence_ratio=0.732 hz_speed_slope=0.053 offset_R=0.072

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
