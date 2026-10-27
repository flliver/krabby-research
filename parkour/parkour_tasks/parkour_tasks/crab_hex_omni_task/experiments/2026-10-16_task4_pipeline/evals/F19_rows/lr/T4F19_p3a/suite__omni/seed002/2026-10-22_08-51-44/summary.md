```
=== crab-hex gait eval ===
scenario   : suite__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_03-27-53_seed19_X07_p3a_A15pB/model_22993.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7401590785381814  p25=0.6986867776296963  p75=0.7802451251336273
tippy_tap_fraction median=0.44
slip_ratio         median=0.22967250642606846
tracking_ratio     median=0.9676584271669578  (achieved/commanded vx, walking holds; n=700)
schedule_completion_rate=1.0
terminations={'schedule_complete': 1000}
facing_error_rms   median=0.060968814875570876  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.0 (n=307)
     stand: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.8084673810584997 (n=100)
       fwd: cmd 0.25 -> achieved 0.250 m/s | tripod median=0.7445021511206396 (n=100)
     lat_l: cmd 0.22 -> achieved 0.209 m/s | tripod median=0.7314385887810154 (n=100)
     lat_r: cmd 0.22 -> achieved 0.243 m/s | tripod median=0.7540030117918901 (n=100)
      back: cmd 0.22 -> achieved 0.226 m/s | tripod median=0.7717321123643257 (n=100)
      diag: cmd 0.25 -> achieved 0.241 m/s | tripod median=0.7058351431758246 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.001 m/s | tripod median=0.8136577471029858 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.8069795185223996 (n=100)
  move_turn: cmd 0.22 -> achieved 0.166 m/s | tripod median=0.7005155594353714 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.133 m/s | tripod median=0.682957340150103 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.011 initial=0.007 final=0.013 max=0.023 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.001 yaw_rate_ratio=n/a
     stand: err rms=0.023 initial=0.012 final=0.020 max=0.056 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.054 initial=0.014 final=0.086 max=0.116 closed=n/a | settle=0.000 (n=98/100) | progress=-0.007 net=-0.003 yaw_rate_ratio=0.279
     lat_l: err rms=0.020 initial=0.013 final=0.017 max=0.048 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
     lat_r: err rms=0.016 initial=0.013 final=0.013 max=0.036 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
      back: err rms=0.081 initial=0.013 final=0.121 max=0.152 closed=n/a | settle=8.430 (n=92/100) | progress=-0.002 net=0.009 yaw_rate_ratio=0.298
      diag: err rms=0.029 initial=0.015 final=0.021 max=0.072 closed=n/a | settle=0.000 (n=100/100) | progress=0.003 net=0.002 yaw_rate_ratio=0.409
  turn_l90: err rms=0.821 initial=1.566 final=0.110 max=1.576 closed=0.929 | settle=9.620 (n=92/100) | progress=0.147 net=0.121 yaw_rate_ratio=0.597
  turn_r90: err rms=0.832 initial=1.576 final=0.110 max=1.581 closed=0.930 | settle=9.900 (n=88/100) | progress=0.143 net=0.122 yaw_rate_ratio=0.588
  move_turn: err rms=1.085 initial=1.565 final=0.478 max=1.580 closed=0.696 | settle=n/a (n=0/100) | progress=0.091 net=0.091 yaw_rate_ratio=0.386
  move_turn_r: err rms=1.181 initial=1.580 final=0.796 max=1.584 closed=0.496 | settle=n/a (n=0/100) | progress=0.065 net=0.065 yaw_rate_ratio=0.436

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.003 tripod=0.808 | closed=n/a progress=n/a net=0.001 | drift=0.041 dir_err=n/a
        fwd: n=100 surv=1.000 (fwd=1.00) | ratio=0.999 achieved=0.250 tripod=0.745 | closed=n/a progress=-0.007 net=-0.003 | drift=0.251 dir_err=0.037
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=1.025 achieved=0.226 tripod=0.747 | closed=n/a progress=n/a net=0.001 | drift=0.225 dir_err=0.013
       back: n=100 surv=1.000 (back=1.00) | ratio=1.028 achieved=0.226 tripod=0.772 | closed=n/a progress=-0.002 net=0.009 | drift=0.228 dir_err=0.050
       diag: n=100 surv=1.000 (diag=1.00) | ratio=0.963 achieved=0.241 tripod=0.706 | closed=n/a progress=0.003 net=0.002 | drift=0.241 dir_err=0.014
       turn: n=200 surv=1.000 (turn_l90=1.00 turn_r90=1.00) | ratio=n/a achieved=0.001 tripod=0.810 | closed=0.929 progress=0.145 net=0.122 | drift=0.050 dir_err=n/a
  move_turn: n=200 surv=1.000 (move_turn=1.00 move_turn_r=1.00) | ratio=0.694 achieved=0.153 tripod=0.689 | closed=0.611 progress=0.080 net=0.080 | drift=0.169 dir_err=0.162

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.187 action_fft_freq=1.778 joint_speed_mean=0.337 rod_speed_mean=0.029 contact_switch_hz=1.500 foot_slip=0.072 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.003 settle_six_down_s=12.000 foot_disp_end=0.215 work_per_s=250.470
        fwd: action_jitter=0.157 action_fft_freq=1.344 joint_speed_mean=0.716 rod_speed_mean=0.029 contact_switch_hz=1.875 foot_slip=0.094 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=22.715 slip_per_m=1.119 work_per_m=1042.990 cam_work_per_m=393.790
       side: action_jitter=0.127 action_fft_freq=1.352 joint_speed_mean=0.361 rod_speed_mean=0.028 contact_switch_hz=1.486 foot_slip=0.070 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=20.416 slip_per_m=0.997 work_per_m=1005.644 cam_work_per_m=174.312
       back: action_jitter=0.157 action_fft_freq=1.350 joint_speed_mean=0.679 rod_speed_mean=0.029 contact_switch_hz=1.833 foot_slip=0.087 wz_active_frac=0.102 clock_restarts=0.000 swings_per_m=24.181 slip_per_m=1.142 work_per_m=1042.195 cam_work_per_m=428.609
       diag: action_jitter=0.140 action_fft_freq=1.357 joint_speed_mean=0.598 rod_speed_mean=0.029 contact_switch_hz=1.924 foot_slip=0.090 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=24.499 slip_per_m=1.140 work_per_m=976.062 cam_work_per_m=295.461
       turn: action_jitter=0.151 action_fft_freq=1.245 joint_speed_mean=0.683 rod_speed_mean=0.029 contact_switch_hz=1.556 foot_slip=0.079 wz_active_frac=0.826 clock_restarts=1.000 swings_per_rad=38.475 slip_per_rad=1.992 work_per_rad=1903.417 cam_work_per_rad=623.968
  move_turn: action_jitter=0.137 action_fft_freq=1.249 joint_speed_mean=0.637 rod_speed_mean=0.028 contact_switch_hz=1.722 foot_slip=0.105 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=67.225 slip_per_rad=4.044 work_per_rad=3741.354 cam_work_per_rad=1184.981

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.058 steps_per_s=2.500 gait_hz=0.385 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=10.916 support_feet=2.942 taps_per_m=11.696 swing_apex=0.113 step_rate_hz=2.750 gait_hz=0.403 clock_hz=0.357 cadence_ratio=1.128 hz_speed_slope=0.605 offset_R=0.191
       side: steps_per_m=11.644 support_feet=3.070 taps_per_m=8.404 swing_apex=0.102 step_rate_hz=2.583 gait_hz=0.397 clock_hz=0.314 cadence_ratio=1.262 hz_speed_slope=-0.948 offset_R=0.059
       back: steps_per_m=12.368 support_feet=2.957 taps_per_m=11.707 swing_apex=0.115 step_rate_hz=2.750 gait_hz=0.415 clock_hz=0.319 cadence_ratio=1.288 hz_speed_slope=0.283 offset_R=0.090
       diag: steps_per_m=13.406 support_feet=3.007 taps_per_m=10.980 swing_apex=0.092 step_rate_hz=3.167 gait_hz=0.464 clock_hz=0.357 cadence_ratio=1.299 hz_speed_slope=0.479 offset_R=0.110
       turn: steps_per_rad=21.424 support_feet=3.079 taps_per_rad=17.037 swing_apex=0.124 step_rate_hz=2.583 gait_hz=0.415 clock_hz=0.408 cadence_ratio=1.019 steps_per_rad_pre=17.959 steps_per_rad_post=3.955 offset_R=0.057
  move_turn: steps_per_rad=33.774 support_feet=2.989 taps_per_rad=33.043 swing_apex=0.123 step_rate_hz=2.583 gait_hz=0.372 clock_hz=0.500 cadence_ratio=0.745 hz_speed_slope=0.671 offset_R=0.068

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
