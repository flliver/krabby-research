```
=== crab-hex gait eval ===
scenario   : suite__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7208339679461997  p25=0.6886280859265729  p75=0.7722907167470973
tippy_tap_fraction median=0.4594594594594595
slip_ratio         median=0.2432560054400591
tracking_ratio     median=1.004921762160558  (achieved/commanded vx, walking holds; n=700)
schedule_completion_rate=1.0
terminations={'schedule_complete': 1000}
facing_error_rms   median=0.05034211169001036  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.7687216296988919 (n=975)
     stand: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7908561719938312 (n=100)
       fwd: cmd 0.25 -> achieved 0.252 m/s | tripod median=0.6729687895953607 (n=100)
     lat_l: cmd 0.22 -> achieved 0.235 m/s | tripod median=0.6766726873070987 (n=100)
     lat_r: cmd 0.22 -> achieved 0.232 m/s | tripod median=0.7647424241850644 (n=100)
      back: cmd 0.22 -> achieved 0.218 m/s | tripod median=0.6943996379473288 (n=100)
      diag: cmd 0.25 -> achieved 0.254 m/s | tripod median=0.6938199663751761 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.004 m/s | tripod median=0.8013247180858708 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.006 m/s | tripod median=0.7934679112484814 (n=100)
  move_turn: cmd 0.22 -> achieved 0.159 m/s | tripod median=0.676967333521074 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.167 m/s | tripod median=0.7148182351397308 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.014 initial=0.008 final=0.013 max=0.028 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.002 yaw_rate_ratio=n/a
     stand: err rms=0.023 initial=0.012 final=0.020 max=0.054 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.042 initial=0.013 final=0.030 max=0.096 closed=n/a | settle=0.000 (n=100/100) | progress=-0.000 net=0.002 yaw_rate_ratio=0.383
     lat_l: err rms=0.019 initial=0.010 final=0.017 max=0.043 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
     lat_r: err rms=0.029 initial=0.013 final=0.032 max=0.066 closed=n/a | settle=0.000 (n=100/100) | progress=0.009 net=0.000 yaw_rate_ratio=0.346
      back: err rms=0.059 initial=0.013 final=0.076 max=0.121 closed=n/a | settle=0.000 (n=100/100) | progress=-0.000 net=-0.002 yaw_rate_ratio=0.392
      diag: err rms=0.033 initial=0.013 final=0.036 max=0.076 closed=n/a | settle=0.000 (n=100/100) | progress=-0.005 net=0.001 yaw_rate_ratio=0.369
  turn_l90: err rms=0.784 initial=1.573 final=0.020 max=1.577 closed=0.987 | settle=8.120 (n=100/100) | progress=0.176 net=0.132 yaw_rate_ratio=0.660
  turn_r90: err rms=0.796 initial=1.570 final=0.019 max=1.576 closed=0.988 | settle=8.360 (n=100/100) | progress=0.171 net=0.130 yaw_rate_ratio=0.646
  move_turn: err rms=1.085 initial=1.568 final=0.589 max=1.569 closed=0.627 | settle=n/a (n=0/100) | progress=0.081 net=0.082 yaw_rate_ratio=0.420
  move_turn_r: err rms=1.023 initial=1.571 final=0.359 max=1.579 closed=0.773 | settle=11.960 (n=1/100) | progress=0.101 net=0.101 yaw_rate_ratio=0.454

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.003 tripod=0.791 | closed=n/a progress=n/a net=0.001 | drift=0.034 dir_err=n/a
        fwd: n=100 surv=1.000 (fwd=1.00) | ratio=1.009 achieved=0.252 tripod=0.673 | closed=n/a progress=-0.000 net=0.002 | drift=0.252 dir_err=0.010
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=1.062 achieved=0.234 tripod=0.718 | closed=n/a progress=0.009 net=0.001 | drift=0.234 dir_err=0.018
       back: n=100 surv=1.000 (back=1.00) | ratio=0.993 achieved=0.218 tripod=0.694 | closed=n/a progress=-0.000 net=-0.002 | drift=0.219 dir_err=0.015
       diag: n=100 surv=1.000 (diag=1.00) | ratio=1.014 achieved=0.254 tripod=0.694 | closed=n/a progress=-0.005 net=0.001 | drift=0.257 dir_err=0.065
       turn: n=200 surv=1.000 (turn_l90=1.00 turn_r90=1.00) | ratio=n/a achieved=0.002 tripod=0.798 | closed=0.988 progress=0.174 net=0.131 | drift=0.046 dir_err=n/a
  move_turn: n=200 surv=1.000 (move_turn=1.00 move_turn_r=1.00) | ratio=0.741 achieved=0.163 tripod=0.689 | closed=0.687 progress=0.089 net=0.090 | drift=0.177 dir_err=0.161

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.207 action_fft_freq=1.996 joint_speed_mean=0.366 rod_speed_mean=0.029 contact_switch_hz=1.750 foot_slip=0.073 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.012 settle_six_down_s=12.000 foot_disp_end=0.091 work_per_s=291.953
        fwd: action_jitter=0.132 action_fft_freq=1.151 joint_speed_mean=0.755 rod_speed_mean=0.029 contact_switch_hz=1.944 foot_slip=0.095 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=23.200 slip_per_m=1.102 work_per_m=1088.859 cam_work_per_m=405.198
       side: action_jitter=0.108 action_fft_freq=1.230 joint_speed_mean=0.389 rod_speed_mean=0.029 contact_switch_hz=1.500 foot_slip=0.075 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=19.293 slip_per_m=0.988 work_per_m=1012.815 cam_work_per_m=199.011
       back: action_jitter=0.142 action_fft_freq=1.259 joint_speed_mean=0.706 rod_speed_mean=0.029 contact_switch_hz=1.896 foot_slip=0.092 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=26.493 slip_per_m=1.258 work_per_m=1207.980 cam_work_per_m=393.248
       diag: action_jitter=0.120 action_fft_freq=1.263 joint_speed_mean=0.578 rod_speed_mean=0.029 contact_switch_hz=1.722 foot_slip=0.088 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=20.379 slip_per_m=1.018 work_per_m=1007.419 cam_work_per_m=223.688
       turn: action_jitter=0.153 action_fft_freq=1.281 joint_speed_mean=0.679 rod_speed_mean=0.029 contact_switch_hz=1.667 foot_slip=0.074 wz_active_frac=0.693 clock_restarts=0.000 swings_per_rad=38.501 slip_per_rad=1.760 work_per_rad=1842.052 cam_work_per_rad=594.900
  move_turn: action_jitter=0.120 action_fft_freq=1.119 joint_speed_mean=0.685 rod_speed_mean=0.028 contact_switch_hz=1.931 foot_slip=0.102 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=64.453 slip_per_rad=3.361 work_per_rad=3228.520 cam_work_per_rad=1099.215

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.162 steps_per_s=2.500 gait_hz=0.421 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=10.791 support_feet=2.943 taps_per_m=12.444 swing_apex=0.119 step_rate_hz=2.708 gait_hz=0.397 clock_hz=0.357 cadence_ratio=1.111 hz_speed_slope=-0.068 offset_R=0.162
       side: steps_per_m=11.146 support_feet=3.067 taps_per_m=8.110 swing_apex=0.099 step_rate_hz=2.583 gait_hz=0.421 clock_hz=0.314 cadence_ratio=1.340 hz_speed_slope=3.011 offset_R=0.056
       back: steps_per_m=12.906 support_feet=2.965 taps_per_m=13.349 swing_apex=0.109 step_rate_hz=2.750 gait_hz=0.421 clock_hz=0.314 cadence_ratio=1.340 hz_speed_slope=0.168 offset_R=0.039
       diag: steps_per_m=11.756 support_feet=2.945 taps_per_m=8.353 swing_apex=0.095 step_rate_hz=3.000 gait_hz=0.470 clock_hz=0.357 cadence_ratio=1.316 hz_speed_slope=-0.568 offset_R=0.082
       turn: steps_per_rad=19.654 support_feet=3.100 taps_per_rad=17.932 swing_apex=0.139 step_rate_hz=2.583 gait_hz=0.439 clock_hz=0.345 cadence_ratio=1.259 steps_per_rad_pre=13.567 steps_per_rad_post=5.849 offset_R=0.064
  move_turn: steps_per_rad=30.188 support_feet=3.007 taps_per_rad=33.645 swing_apex=0.111 step_rate_hz=2.750 gait_hz=0.372 clock_hz=0.500 cadence_ratio=0.745 hz_speed_slope=-0.038 offset_R=0.016

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
