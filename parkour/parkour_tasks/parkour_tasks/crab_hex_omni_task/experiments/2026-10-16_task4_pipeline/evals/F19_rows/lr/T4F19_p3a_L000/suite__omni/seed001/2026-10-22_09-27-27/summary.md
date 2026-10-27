```
=== crab-hex gait eval ===
scenario   : suite__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_03-27-53_seed19_X07_p3a_A15pB/model_22993.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7372400760595063  p25=0.6965227107956427  p75=0.7787955314470174
tippy_tap_fraction median=0.43705985915492956
slip_ratio         median=0.23153184515405673
tracking_ratio     median=0.9630451453753051  (achieved/commanded vx, walking holds; n=700)
schedule_completion_rate=1.0
terminations={'schedule_complete': 1000}
facing_error_rms   median=0.06040929162356771  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.0723266581452995 (n=330)
     stand: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.8064698650348501 (n=100)
       fwd: cmd 0.25 -> achieved 0.250 m/s | tripod median=0.7430156962312456 (n=100)
     lat_l: cmd 0.22 -> achieved 0.208 m/s | tripod median=0.7349983734720643 (n=100)
     lat_r: cmd 0.22 -> achieved 0.243 m/s | tripod median=0.7514476445882079 (n=100)
      back: cmd 0.22 -> achieved 0.226 m/s | tripod median=0.768549146585277 (n=100)
      diag: cmd 0.25 -> achieved 0.240 m/s | tripod median=0.7041080733165412 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.002 m/s | tripod median=0.8120977859917788 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.8066087763500149 (n=100)
  move_turn: cmd 0.22 -> achieved 0.166 m/s | tripod median=0.6969093807900659 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.133 m/s | tripod median=0.692142111486303 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.011 initial=0.008 final=0.013 max=0.023 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.001 yaw_rate_ratio=n/a
     stand: err rms=0.024 initial=0.010 final=0.018 max=0.061 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.052 initial=0.015 final=0.090 max=0.111 closed=n/a | settle=0.000 (n=97/100) | progress=-0.008 net=-0.003 yaw_rate_ratio=0.263
     lat_l: err rms=0.019 initial=0.014 final=0.017 max=0.047 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.002 yaw_rate_ratio=n/a
     lat_r: err rms=0.016 initial=0.014 final=0.016 max=0.038 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
      back: err rms=0.080 initial=0.013 final=0.119 max=0.152 closed=n/a | settle=0.000 (n=94/100) | progress=-0.002 net=0.007 yaw_rate_ratio=0.304
      diag: err rms=0.029 initial=0.014 final=0.020 max=0.072 closed=n/a | settle=0.000 (n=100/100) | progress=0.001 net=0.001 yaw_rate_ratio=0.388
  turn_l90: err rms=0.817 initial=1.563 final=0.114 max=1.570 closed=0.927 | settle=9.700 (n=94/100) | progress=0.146 net=0.121 yaw_rate_ratio=0.595
  turn_r90: err rms=0.832 initial=1.577 final=0.113 max=1.584 closed=0.928 | settle=10.290 (n=94/100) | progress=0.141 net=0.122 yaw_rate_ratio=0.590
  move_turn: err rms=1.082 initial=1.566 final=0.479 max=1.577 closed=0.696 | settle=n/a (n=0/100) | progress=0.090 net=0.091 yaw_rate_ratio=0.385
  move_turn_r: err rms=1.167 initial=1.577 final=0.806 max=1.583 closed=0.492 | settle=n/a (n=0/100) | progress=0.064 net=0.065 yaw_rate_ratio=0.439

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.003 tripod=0.806 | closed=n/a progress=n/a net=0.001 | drift=0.041 dir_err=n/a
        fwd: n=100 surv=1.000 (fwd=1.00) | ratio=1.000 achieved=0.250 tripod=0.743 | closed=n/a progress=-0.008 net=-0.003 | drift=0.252 dir_err=0.037
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=1.019 achieved=0.224 tripod=0.745 | closed=n/a progress=n/a net=0.001 | drift=0.224 dir_err=0.011
       back: n=100 surv=1.000 (back=1.00) | ratio=1.029 achieved=0.226 tripod=0.769 | closed=n/a progress=-0.002 net=0.007 | drift=0.228 dir_err=0.049
       diag: n=100 surv=1.000 (diag=1.00) | ratio=0.959 achieved=0.240 tripod=0.704 | closed=n/a progress=0.001 net=0.001 | drift=0.241 dir_err=0.017
       turn: n=200 surv=1.000 (turn_l90=1.00 turn_r90=1.00) | ratio=n/a achieved=0.001 tripod=0.809 | closed=0.928 progress=0.143 net=0.121 | drift=0.050 dir_err=n/a
  move_turn: n=200 surv=1.000 (move_turn=1.00 move_turn_r=1.00) | ratio=0.687 achieved=0.151 tripod=0.694 | closed=0.591 progress=0.077 net=0.077 | drift=0.167 dir_err=0.152

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.187 action_fft_freq=1.779 joint_speed_mean=0.335 rod_speed_mean=0.029 contact_switch_hz=1.444 foot_slip=0.071 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.003 settle_six_down_s=12.000 foot_disp_end=0.215 work_per_s=249.281
        fwd: action_jitter=0.156 action_fft_freq=1.339 joint_speed_mean=0.712 rod_speed_mean=0.029 contact_switch_hz=1.847 foot_slip=0.094 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=22.336 slip_per_m=1.117 work_per_m=1052.514 cam_work_per_m=401.446
       side: action_jitter=0.129 action_fft_freq=1.352 joint_speed_mean=0.364 rod_speed_mean=0.028 contact_switch_hz=1.465 foot_slip=0.071 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=20.342 slip_per_m=1.003 work_per_m=1009.613 cam_work_per_m=165.085
       back: action_jitter=0.157 action_fft_freq=1.346 joint_speed_mean=0.680 rod_speed_mean=0.029 contact_switch_hz=1.833 foot_slip=0.088 wz_active_frac=0.087 clock_restarts=0.000 swings_per_m=24.467 slip_per_m=1.155 work_per_m=1053.236 cam_work_per_m=429.125
       diag: action_jitter=0.140 action_fft_freq=1.355 joint_speed_mean=0.599 rod_speed_mean=0.029 contact_switch_hz=1.944 foot_slip=0.089 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=24.724 slip_per_m=1.138 work_per_m=984.267 cam_work_per_m=302.808
       turn: action_jitter=0.151 action_fft_freq=1.248 joint_speed_mean=0.683 rod_speed_mean=0.029 contact_switch_hz=1.583 foot_slip=0.078 wz_active_frac=0.842 clock_restarts=1.000 swings_per_rad=38.901 slip_per_rad=1.961 work_per_rad=1917.004 cam_work_per_rad=612.925
  move_turn: action_jitter=0.138 action_fft_freq=1.250 joint_speed_mean=0.637 rod_speed_mean=0.028 contact_switch_hz=1.715 foot_slip=0.105 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=66.815 slip_per_rad=3.964 work_per_rad=3708.066 cam_work_per_rad=1308.374

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.063 steps_per_s=2.500 gait_hz=0.391 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=10.867 support_feet=2.951 taps_per_m=11.129 swing_apex=0.113 step_rate_hz=2.750 gait_hz=0.403 clock_hz=0.357 cadence_ratio=1.128 hz_speed_slope=0.214 offset_R=0.110
       side: steps_per_m=11.499 support_feet=3.088 taps_per_m=8.133 swing_apex=0.102 step_rate_hz=2.583 gait_hz=0.397 clock_hz=0.314 cadence_ratio=1.262 hz_speed_slope=-0.895 offset_R=0.077
       back: steps_per_m=12.492 support_feet=2.957 taps_per_m=11.681 swing_apex=0.114 step_rate_hz=2.833 gait_hz=0.415 clock_hz=0.319 cadence_ratio=1.295 hz_speed_slope=0.241 offset_R=0.138
       diag: steps_per_m=13.374 support_feet=3.013 taps_per_m=11.178 swing_apex=0.091 step_rate_hz=3.167 gait_hz=0.464 clock_hz=0.357 cadence_ratio=1.299 hz_speed_slope=-0.209 offset_R=0.039
       turn: steps_per_rad=21.376 support_feet=3.081 taps_per_rad=17.052 swing_apex=0.124 step_rate_hz=2.583 gait_hz=0.415 clock_hz=0.411 cadence_ratio=1.014 steps_per_rad_pre=17.791 steps_per_rad_post=3.975 offset_R=0.038
  move_turn: steps_per_rad=34.158 support_feet=2.980 taps_per_rad=32.445 swing_apex=0.123 step_rate_hz=2.625 gait_hz=0.372 clock_hz=0.500 cadence_ratio=0.745 hz_speed_slope=0.686 offset_R=0.016

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
