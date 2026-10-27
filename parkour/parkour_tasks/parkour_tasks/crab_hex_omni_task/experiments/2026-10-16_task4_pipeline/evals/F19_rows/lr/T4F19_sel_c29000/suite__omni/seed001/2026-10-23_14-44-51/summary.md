```
=== crab-hex gait eval ===
scenario   : suite__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_29000.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7123910301191809  p25=0.6703341091452655  p75=0.7617242361402018
tippy_tap_fraction median=0.4263500482160077
slip_ratio         median=0.23264743608312366
tracking_ratio     median=0.9986150620059557  (achieved/commanded vx, walking holds; n=700)
schedule_completion_rate=1.0
terminations={'schedule_complete': 1000}
facing_error_rms   median=0.053679881712997216  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.709875524972907 (n=985)
     stand: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7697145977711779 (n=100)
       fwd: cmd 0.25 -> achieved 0.251 m/s | tripod median=0.6691274296551187 (n=100)
     lat_l: cmd 0.22 -> achieved 0.229 m/s | tripod median=0.642153226330481 (n=100)
     lat_r: cmd 0.22 -> achieved 0.230 m/s | tripod median=0.7701884945416486 (n=100)
      back: cmd 0.22 -> achieved 0.217 m/s | tripod median=0.7078924760986647 (n=100)
      diag: cmd 0.25 -> achieved 0.252 m/s | tripod median=0.691417778208089 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.003 m/s | tripod median=0.794451770963505 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.006 m/s | tripod median=0.7858932103581313 (n=100)
  move_turn: cmd 0.22 -> achieved 0.162 m/s | tripod median=0.643624201090053 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.164 m/s | tripod median=0.7127311045212301 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.014 initial=0.007 final=0.015 max=0.029 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.001 yaw_rate_ratio=n/a
     stand: err rms=0.030 initial=0.012 final=0.034 max=0.062 closed=n/a | settle=0.000 (n=100/100) | progress=-0.264 net=-0.001 yaw_rate_ratio=1.283
       fwd: err rms=0.043 initial=0.014 final=0.041 max=0.103 closed=n/a | settle=0.000 (n=100/100) | progress=-0.002 net=0.002 yaw_rate_ratio=0.523
     lat_l: err rms=0.022 initial=0.013 final=0.023 max=0.053 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.002 yaw_rate_ratio=n/a
     lat_r: err rms=0.035 initial=0.017 final=0.032 max=0.073 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.002 yaw_rate_ratio=n/a
      back: err rms=0.069 initial=0.014 final=0.086 max=0.130 closed=n/a | settle=0.000 (n=100/100) | progress=-0.000 net=-0.004 yaw_rate_ratio=0.507
      diag: err rms=0.038 initial=0.018 final=0.051 max=0.087 closed=n/a | settle=0.000 (n=100/100) | progress=-0.001 net=0.002 yaw_rate_ratio=0.338
  turn_l90: err rms=0.784 initial=1.568 final=0.017 max=1.573 closed=0.989 | settle=8.220 (n=100/100) | progress=0.174 net=0.131 yaw_rate_ratio=0.668
  turn_r90: err rms=0.807 initial=1.584 final=0.020 max=1.584 closed=0.987 | settle=8.450 (n=100/100) | progress=0.170 net=0.130 yaw_rate_ratio=0.649
  move_turn: err rms=1.107 initial=1.565 final=0.644 max=1.571 closed=0.587 | settle=n/a (n=0/100) | progress=0.076 net=0.077 yaw_rate_ratio=0.415
  move_turn_r: err rms=1.012 initial=1.577 final=0.341 max=1.581 closed=0.783 | settle=11.860 (n=2/100) | progress=0.104 net=0.104 yaw_rate_ratio=0.454

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.003 tripod=0.770 | closed=n/a progress=-0.264 net=-0.001 | drift=0.034 dir_err=n/a
        fwd: n=100 surv=1.000 (fwd=1.00) | ratio=1.005 achieved=0.251 tripod=0.669 | closed=n/a progress=-0.002 net=0.002 | drift=0.251 dir_err=0.012
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=1.044 achieved=0.230 tripod=0.706 | closed=n/a progress=n/a net=0.002 | drift=0.231 dir_err=0.027
       back: n=100 surv=1.000 (back=1.00) | ratio=0.985 achieved=0.217 tripod=0.708 | closed=n/a progress=-0.000 net=-0.004 | drift=0.217 dir_err=0.015
       diag: n=100 surv=1.000 (diag=1.00) | ratio=1.006 achieved=0.252 tripod=0.691 | closed=n/a progress=-0.001 net=0.002 | drift=0.256 dir_err=0.055
       turn: n=200 surv=1.000 (turn_l90=1.00 turn_r90=1.00) | ratio=n/a achieved=0.001 tripod=0.792 | closed=0.989 progress=0.172 net=0.130 | drift=0.047 dir_err=n/a
  move_turn: n=200 surv=1.000 (move_turn=1.00 move_turn_r=1.00) | ratio=0.740 achieved=0.163 tripod=0.678 | closed=0.699 progress=0.090 net=0.091 | drift=0.176 dir_err=0.155

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.143 action_fft_freq=1.630 joint_speed_mean=0.367 rod_speed_mean=0.029 contact_switch_hz=1.639 foot_slip=0.074 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.012 settle_six_down_s=12.000 foot_disp_end=0.103 work_per_s=293.413
        fwd: action_jitter=0.102 action_fft_freq=1.055 joint_speed_mean=0.759 rod_speed_mean=0.029 contact_switch_hz=1.833 foot_slip=0.097 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=21.889 slip_per_m=1.140 work_per_m=1122.594 cam_work_per_m=415.818
       side: action_jitter=0.083 action_fft_freq=1.137 joint_speed_mean=0.387 rod_speed_mean=0.029 contact_switch_hz=1.444 foot_slip=0.073 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=18.806 slip_per_m=0.973 work_per_m=1036.301 cam_work_per_m=201.062
       back: action_jitter=0.116 action_fft_freq=1.164 joint_speed_mean=0.700 rod_speed_mean=0.029 contact_switch_hz=1.799 foot_slip=0.090 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=24.957 slip_per_m=1.248 work_per_m=1197.723 cam_work_per_m=405.587
       diag: action_jitter=0.096 action_fft_freq=1.178 joint_speed_mean=0.578 rod_speed_mean=0.029 contact_switch_hz=1.694 foot_slip=0.088 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=20.063 slip_per_m=1.034 work_per_m=1015.047 cam_work_per_m=211.943
       turn: action_jitter=0.110 action_fft_freq=1.113 joint_speed_mean=0.682 rod_speed_mean=0.029 contact_switch_hz=1.556 foot_slip=0.075 wz_active_frac=0.700 clock_restarts=0.000 swings_per_rad=35.861 slip_per_rad=1.785 work_per_rad=1885.024 cam_work_per_rad=612.488
  move_turn: action_jitter=0.094 action_fft_freq=1.049 joint_speed_mean=0.679 rod_speed_mean=0.028 contact_switch_hz=1.861 foot_slip=0.099 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=61.587 slip_per_rad=3.378 work_per_rad=3221.656 cam_work_per_rad=1127.360

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.199 steps_per_s=2.417 gait_hz=0.409 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=10.542 support_feet=2.971 taps_per_m=11.265 swing_apex=0.120 step_rate_hz=2.667 gait_hz=0.397 clock_hz=0.357 cadence_ratio=1.109 hz_speed_slope=-0.141 offset_R=0.143
       side: steps_per_m=11.276 support_feet=3.047 taps_per_m=7.290 swing_apex=0.102 step_rate_hz=2.583 gait_hz=0.412 clock_hz=0.314 cadence_ratio=1.311 hz_speed_slope=-1.231 offset_R=0.017
       back: steps_per_m=12.842 support_feet=2.984 taps_per_m=12.204 swing_apex=0.106 step_rate_hz=2.750 gait_hz=0.427 clock_hz=0.314 cadence_ratio=1.359 hz_speed_slope=0.150 offset_R=0.110
       diag: steps_per_m=12.285 support_feet=2.957 taps_per_m=7.642 swing_apex=0.096 step_rate_hz=3.083 gait_hz=0.470 clock_hz=0.357 cadence_ratio=1.316 hz_speed_slope=-1.179 offset_R=0.054
       turn: steps_per_rad=19.646 support_feet=3.112 taps_per_rad=15.974 swing_apex=0.142 step_rate_hz=2.583 gait_hz=0.427 clock_hz=0.348 cadence_ratio=1.243 steps_per_rad_pre=13.765 steps_per_rad_post=5.785 offset_R=0.007
  move_turn: steps_per_rad=29.594 support_feet=3.018 taps_per_rad=31.375 swing_apex=0.105 step_rate_hz=2.750 gait_hz=0.366 clock_hz=0.500 cadence_ratio=0.732 hz_speed_slope=0.211 offset_R=0.061

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
