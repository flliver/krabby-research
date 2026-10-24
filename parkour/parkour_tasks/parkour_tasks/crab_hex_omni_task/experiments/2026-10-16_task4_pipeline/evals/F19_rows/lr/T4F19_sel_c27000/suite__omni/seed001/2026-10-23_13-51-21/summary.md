```
=== crab-hex gait eval ===
scenario   : suite__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_27000.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6991231585965479  p25=0.6584524803218492  p75=0.7682698634490359
tippy_tap_fraction median=0.4186046511627907
slip_ratio         median=0.2304504740999677
tracking_ratio     median=0.9905280007633152  (achieved/commanded vx, walking holds; n=700)
schedule_completion_rate=1.0
terminations={'schedule_complete': 1000}
facing_error_rms   median=0.05202340463093934  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.7440124592739494 (n=995)
     stand: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7698499831297315 (n=100)
       fwd: cmd 0.25 -> achieved 0.249 m/s | tripod median=0.6443706713939785 (n=100)
     lat_l: cmd 0.22 -> achieved 0.229 m/s | tripod median=0.6301650867942041 (n=100)
     lat_r: cmd 0.22 -> achieved 0.229 m/s | tripod median=0.7675474686155643 (n=100)
      back: cmd 0.22 -> achieved 0.214 m/s | tripod median=0.6764737210324847 (n=100)
      diag: cmd 0.25 -> achieved 0.251 m/s | tripod median=0.6772433588362545 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.002 m/s | tripod median=0.7970270805983011 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.006 m/s | tripod median=0.786543788131257 (n=100)
  move_turn: cmd 0.22 -> achieved 0.162 m/s | tripod median=0.6399955693755885 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.162 m/s | tripod median=0.6936363024314643 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.014 initial=0.007 final=0.013 max=0.029 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.001 yaw_rate_ratio=n/a
     stand: err rms=0.023 initial=0.012 final=0.027 max=0.056 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.044 initial=0.014 final=0.046 max=0.102 closed=n/a | settle=0.000 (n=100/100) | progress=-0.001 net=0.003 yaw_rate_ratio=0.727
     lat_l: err rms=0.022 initial=0.014 final=0.019 max=0.051 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.002 yaw_rate_ratio=n/a
     lat_r: err rms=0.030 initial=0.013 final=0.032 max=0.064 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
      back: err rms=0.062 initial=0.016 final=0.087 max=0.130 closed=n/a | settle=0.000 (n=100/100) | progress=-0.003 net=-0.003 yaw_rate_ratio=0.535
      diag: err rms=0.033 initial=0.014 final=0.023 max=0.076 closed=n/a | settle=0.000 (n=97/100) | progress=-0.001 net=0.001 yaw_rate_ratio=0.331
  turn_l90: err rms=0.788 initial=1.571 final=0.019 max=1.573 closed=0.988 | settle=8.290 (n=100/100) | progress=0.170 net=0.130 yaw_rate_ratio=0.659
  turn_r90: err rms=0.801 initial=1.578 final=0.020 max=1.580 closed=0.987 | settle=8.320 (n=100/100) | progress=0.172 net=0.131 yaw_rate_ratio=0.649
  move_turn: err rms=1.076 initial=1.564 final=0.562 max=1.567 closed=0.641 | settle=n/a (n=0/100) | progress=0.083 net=0.084 yaw_rate_ratio=0.409
  move_turn_r: err rms=1.027 initial=1.577 final=0.370 max=1.583 closed=0.765 | settle=n/a (n=0/100) | progress=0.101 net=0.101 yaw_rate_ratio=0.454

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.003 tripod=0.770 | closed=n/a progress=n/a net=0.001 | drift=0.035 dir_err=n/a
        fwd: n=100 surv=1.000 (fwd=1.00) | ratio=0.995 achieved=0.249 tripod=0.644 | closed=n/a progress=-0.001 net=0.003 | drift=0.249 dir_err=0.009
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=1.043 achieved=0.229 tripod=0.693 | closed=n/a progress=n/a net=0.001 | drift=0.231 dir_err=0.025
       back: n=100 surv=1.000 (back=1.00) | ratio=0.973 achieved=0.214 tripod=0.676 | closed=n/a progress=-0.003 net=-0.003 | drift=0.215 dir_err=0.039
       diag: n=100 surv=1.000 (diag=1.00) | ratio=1.004 achieved=0.251 tripod=0.677 | closed=n/a progress=-0.001 net=0.001 | drift=0.255 dir_err=0.066
       turn: n=200 surv=1.000 (turn_l90=1.00 turn_r90=1.00) | ratio=n/a achieved=0.002 tripod=0.792 | closed=0.988 progress=0.171 net=0.130 | drift=0.047 dir_err=n/a
  move_turn: n=200 surv=1.000 (move_turn=1.00 move_turn_r=1.00) | ratio=0.736 achieved=0.162 tripod=0.668 | closed=0.701 progress=0.091 net=0.091 | drift=0.174 dir_err=0.154

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.146 action_fft_freq=1.680 joint_speed_mean=0.362 rod_speed_mean=0.029 contact_switch_hz=1.667 foot_slip=0.073 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.010 settle_six_down_s=12.000 foot_disp_end=0.095 work_per_s=288.539
        fwd: action_jitter=0.103 action_fft_freq=1.052 joint_speed_mean=0.750 rod_speed_mean=0.029 contact_switch_hz=1.917 foot_slip=0.096 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=23.033 slip_per_m=1.166 work_per_m=1137.841 cam_work_per_m=424.615
       side: action_jitter=0.086 action_fft_freq=1.172 joint_speed_mean=0.384 rod_speed_mean=0.029 contact_switch_hz=1.444 foot_slip=0.075 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=18.988 slip_per_m=0.997 work_per_m=1036.258 cam_work_per_m=199.346
       back: action_jitter=0.112 action_fft_freq=1.155 joint_speed_mean=0.703 rod_speed_mean=0.029 contact_switch_hz=1.861 foot_slip=0.091 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=26.454 slip_per_m=1.274 work_per_m=1226.880 cam_work_per_m=420.864
       diag: action_jitter=0.098 action_fft_freq=1.187 joint_speed_mean=0.576 rod_speed_mean=0.029 contact_switch_hz=1.750 foot_slip=0.087 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=20.951 slip_per_m=1.024 work_per_m=1013.133 cam_work_per_m=208.914
       turn: action_jitter=0.111 action_fft_freq=1.136 joint_speed_mean=0.670 rod_speed_mean=0.029 contact_switch_hz=1.639 foot_slip=0.075 wz_active_frac=0.702 clock_restarts=0.000 swings_per_rad=37.396 slip_per_rad=1.795 work_per_rad=1834.419 cam_work_per_rad=618.565
  move_turn: action_jitter=0.093 action_fft_freq=1.055 joint_speed_mean=0.674 rod_speed_mean=0.028 contact_switch_hz=1.917 foot_slip=0.098 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=62.151 slip_per_rad=3.274 work_per_rad=3112.698 cam_work_per_rad=1077.375

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.201 steps_per_s=2.583 gait_hz=0.433 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=11.363 support_feet=2.970 taps_per_m=11.649 swing_apex=0.113 step_rate_hz=2.833 gait_hz=0.403 clock_hz=0.357 cadence_ratio=1.128 hz_speed_slope=-0.268 offset_R=0.013
       side: steps_per_m=11.933 support_feet=3.045 taps_per_m=7.252 swing_apex=0.097 step_rate_hz=2.750 gait_hz=0.418 clock_hz=0.314 cadence_ratio=1.330 hz_speed_slope=-0.259 offset_R=0.066
       back: steps_per_m=13.164 support_feet=2.966 taps_per_m=13.362 swing_apex=0.099 step_rate_hz=2.833 gait_hz=0.427 clock_hz=0.314 cadence_ratio=1.359 hz_speed_slope=0.307 offset_R=0.054
       diag: steps_per_m=12.923 support_feet=2.973 taps_per_m=7.791 swing_apex=0.092 step_rate_hz=3.250 gait_hz=0.482 clock_hz=0.357 cadence_ratio=1.350 hz_speed_slope=-0.099 offset_R=0.062
       turn: steps_per_rad=20.669 support_feet=3.118 taps_per_rad=16.626 swing_apex=0.134 step_rate_hz=2.708 gait_hz=0.458 clock_hz=0.349 cadence_ratio=1.290 steps_per_rad_pre=14.124 steps_per_rad_post=5.896 offset_R=0.039
  move_turn: steps_per_rad=31.094 support_feet=3.013 taps_per_rad=31.096 swing_apex=0.100 step_rate_hz=2.833 gait_hz=0.372 clock_hz=0.500 cadence_ratio=0.745 hz_speed_slope=0.107 offset_R=0.084

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
