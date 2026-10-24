```
=== crab-hex gait eval ===
scenario   : suite__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_30991.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7037124794392753  p25=0.6697634722348733  p75=0.7608240501020054
tippy_tap_fraction median=0.4189189189189189
slip_ratio         median=0.2252927268172762
tracking_ratio     median=1.0133566871281428  (achieved/commanded vx, walking holds; n=700)
schedule_completion_rate=1.0
terminations={'schedule_complete': 1000}
facing_error_rms   median=0.054486618167101666  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.008 m/s | tripod median=0.6938525039043975 (n=955)
     stand: cmd 0.00 -> achieved 0.008 m/s | tripod median=0.7624320458500021 (n=100)
       fwd: cmd 0.25 -> achieved 0.256 m/s | tripod median=0.664594817080626 (n=100)
     lat_l: cmd 0.22 -> achieved 0.231 m/s | tripod median=0.6613159728957188 (n=100)
     lat_r: cmd 0.22 -> achieved 0.232 m/s | tripod median=0.7691182242502674 (n=100)
      back: cmd 0.22 -> achieved 0.210 m/s | tripod median=0.7085469685399318 (n=100)
      diag: cmd 0.25 -> achieved 0.257 m/s | tripod median=0.6897211399176364 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.002 m/s | tripod median=0.7994867856841594 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.012 m/s | tripod median=0.7845713656398722 (n=100)
  move_turn: cmd 0.22 -> achieved 0.164 m/s | tripod median=0.657273508321724 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.165 m/s | tripod median=0.6904625150549406 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.014 initial=0.007 final=0.015 max=0.029 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.000 yaw_rate_ratio=n/a
     stand: err rms=0.031 initial=0.014 final=0.036 max=0.071 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.000 yaw_rate_ratio=n/a
       fwd: err rms=0.043 initial=0.013 final=0.037 max=0.098 closed=n/a | settle=0.000 (n=99/100) | progress=-0.002 net=0.002 yaw_rate_ratio=0.498
     lat_l: err rms=0.022 initial=0.017 final=0.028 max=0.049 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.003 yaw_rate_ratio=n/a
     lat_r: err rms=0.034 initial=0.012 final=0.029 max=0.069 closed=n/a | settle=0.000 (n=100/100) | progress=0.005 net=-0.000 yaw_rate_ratio=0.485
      back: err rms=0.070 initial=0.014 final=0.085 max=0.137 closed=n/a | settle=0.000 (n=100/100) | progress=-0.001 net=-0.003 yaw_rate_ratio=0.530
      diag: err rms=0.039 initial=0.014 final=0.049 max=0.083 closed=n/a | settle=0.000 (n=100/100) | progress=-0.002 net=0.001 yaw_rate_ratio=0.352
  turn_l90: err rms=0.780 initial=1.562 final=0.017 max=1.567 closed=0.989 | settle=8.200 (n=100/100) | progress=0.174 net=0.130 yaw_rate_ratio=0.666
  turn_r90: err rms=0.799 initial=1.580 final=0.021 max=1.582 closed=0.986 | settle=8.460 (n=100/100) | progress=0.170 net=0.130 yaw_rate_ratio=0.646
  move_turn: err rms=1.108 initial=1.563 final=0.633 max=1.567 closed=0.592 | settle=n/a (n=0/100) | progress=0.076 net=0.077 yaw_rate_ratio=0.402
  move_turn_r: err rms=1.013 initial=1.584 final=0.361 max=1.587 closed=0.769 | settle=n/a (n=0/100) | progress=0.102 net=0.102 yaw_rate_ratio=0.453

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.008 tripod=0.762 | closed=n/a progress=n/a net=-0.000 | drift=0.036 dir_err=n/a
        fwd: n=100 surv=1.000 (fwd=1.00) | ratio=1.023 achieved=0.256 tripod=0.665 | closed=n/a progress=-0.002 net=0.002 | drift=0.255 dir_err=0.008
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=1.053 achieved=0.232 tripod=0.725 | closed=n/a progress=0.005 net=0.001 | drift=0.233 dir_err=0.046
       back: n=100 surv=1.000 (back=1.00) | ratio=0.954 achieved=0.210 tripod=0.709 | closed=n/a progress=-0.001 net=-0.003 | drift=0.211 dir_err=0.032
       diag: n=100 surv=1.000 (diag=1.00) | ratio=1.029 achieved=0.257 tripod=0.690 | closed=n/a progress=-0.002 net=0.001 | drift=0.261 dir_err=0.055
       turn: n=200 surv=1.000 (turn_l90=1.00 turn_r90=1.00) | ratio=n/a achieved=0.005 tripod=0.789 | closed=0.988 progress=0.172 net=0.130 | drift=0.049 dir_err=n/a
  move_turn: n=200 surv=1.000 (move_turn=1.00 move_turn_r=1.00) | ratio=0.747 achieved=0.164 tripod=0.673 | closed=0.690 progress=0.089 net=0.090 | drift=0.178 dir_err=0.152

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.145 action_fft_freq=1.653 joint_speed_mean=0.368 rod_speed_mean=0.029 contact_switch_hz=1.701 foot_slip=0.074 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.010 settle_six_down_s=12.000 foot_disp_end=0.136 work_per_s=297.929
        fwd: action_jitter=0.103 action_fft_freq=1.052 joint_speed_mean=0.765 rod_speed_mean=0.029 contact_switch_hz=1.806 foot_slip=0.094 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=21.298 slip_per_m=1.086 work_per_m=1103.899 cam_work_per_m=408.195
       side: action_jitter=0.082 action_fft_freq=1.139 joint_speed_mean=0.385 rod_speed_mean=0.029 contact_switch_hz=1.417 foot_slip=0.073 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=18.340 slip_per_m=0.973 work_per_m=1032.928 cam_work_per_m=185.868
       back: action_jitter=0.116 action_fft_freq=1.162 joint_speed_mean=0.693 rod_speed_mean=0.029 contact_switch_hz=1.806 foot_slip=0.090 wz_active_frac=0.009 clock_restarts=0.000 swings_per_m=26.100 slip_per_m=1.282 work_per_m=1239.793 cam_work_per_m=409.212
       diag: action_jitter=0.095 action_fft_freq=1.170 joint_speed_mean=0.581 rod_speed_mean=0.029 contact_switch_hz=1.694 foot_slip=0.087 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=19.622 slip_per_m=1.001 work_per_m=1001.959 cam_work_per_m=216.145
       turn: action_jitter=0.111 action_fft_freq=1.115 joint_speed_mean=0.682 rod_speed_mean=0.029 contact_switch_hz=1.528 foot_slip=0.076 wz_active_frac=0.698 clock_restarts=0.000 swings_per_rad=35.225 slip_per_rad=1.830 work_per_rad=1907.117 cam_work_per_rad=620.791
  move_turn: action_jitter=0.095 action_fft_freq=1.060 joint_speed_mean=0.678 rod_speed_mean=0.028 contact_switch_hz=1.889 foot_slip=0.102 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=64.458 slip_per_rad=3.439 work_per_rad=3209.818 cam_work_per_rad=1130.009

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.187 steps_per_s=2.500 gait_hz=0.403 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=10.315 support_feet=2.959 taps_per_m=10.737 swing_apex=0.119 step_rate_hz=2.667 gait_hz=0.391 clock_hz=0.357 cadence_ratio=1.094 hz_speed_slope=-0.209 offset_R=0.117
       side: steps_per_m=11.171 support_feet=3.046 taps_per_m=6.863 swing_apex=0.104 step_rate_hz=2.583 gait_hz=0.415 clock_hz=0.314 cadence_ratio=1.321 hz_speed_slope=-0.226 offset_R=0.117
       back: steps_per_m=13.463 support_feet=2.965 taps_per_m=12.110 swing_apex=0.106 step_rate_hz=2.750 gait_hz=0.427 clock_hz=0.315 cadence_ratio=1.357 hz_speed_slope=0.147 offset_R=0.082
       diag: steps_per_m=12.119 support_feet=2.953 taps_per_m=7.303 swing_apex=0.091 step_rate_hz=3.167 gait_hz=0.470 clock_hz=0.357 cadence_ratio=1.316 hz_speed_slope=-0.593 offset_R=0.060
       turn: steps_per_rad=19.797 support_feet=3.120 taps_per_rad=15.122 swing_apex=0.142 step_rate_hz=2.583 gait_hz=0.427 clock_hz=0.347 cadence_ratio=1.240 steps_per_rad_pre=13.781 steps_per_rad_post=5.836 offset_R=0.014
  move_turn: steps_per_rad=31.279 support_feet=3.002 taps_per_rad=32.556 swing_apex=0.102 step_rate_hz=2.750 gait_hz=0.366 clock_hz=0.500 cadence_ratio=0.732 hz_speed_slope=0.104 offset_R=0.073

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
