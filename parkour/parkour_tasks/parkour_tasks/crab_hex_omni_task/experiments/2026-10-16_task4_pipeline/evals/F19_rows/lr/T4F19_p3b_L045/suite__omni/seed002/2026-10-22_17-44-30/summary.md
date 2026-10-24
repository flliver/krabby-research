```
=== crab-hex gait eval ===
scenario   : suite__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7061543237014924  p25=0.6715829070498557  p75=0.7640995590665091
tippy_tap_fraction median=0.42857142857142855
slip_ratio         median=0.22779229418916203
tracking_ratio     median=1.016364852436279  (achieved/commanded vx, walking holds; n=700)
schedule_completion_rate=1.0
terminations={'schedule_complete': 1000}
facing_error_rms   median=0.04733425521154096  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.7375423365981578 (n=970)
     stand: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.7812359298482807 (n=100)
       fwd: cmd 0.25 -> achieved 0.255 m/s | tripod median=0.6719780765587222 (n=100)
     lat_l: cmd 0.22 -> achieved 0.235 m/s | tripod median=0.6703705142284595 (n=100)
     lat_r: cmd 0.22 -> achieved 0.234 m/s | tripod median=0.7608699851466465 (n=100)
      back: cmd 0.22 -> achieved 0.221 m/s | tripod median=0.6937065870539689 (n=100)
      diag: cmd 0.25 -> achieved 0.256 m/s | tripod median=0.6873552482621396 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.006 m/s | tripod median=0.7910978365536832 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.007 m/s | tripod median=0.7860341010000448 (n=100)
  move_turn: cmd 0.22 -> achieved 0.167 m/s | tripod median=0.6332563029588579 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.171 m/s | tripod median=0.6924579943281912 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.012 initial=0.007 final=0.011 max=0.026 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.002 yaw_rate_ratio=n/a
     stand: err rms=0.024 initial=0.013 final=0.025 max=0.056 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.040 initial=0.011 final=0.035 max=0.092 closed=n/a | settle=0.000 (n=100/100) | progress=-0.003 net=0.002 yaw_rate_ratio=0.541
     lat_l: err rms=0.018 initial=0.010 final=0.014 max=0.042 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
     lat_r: err rms=0.027 initial=0.012 final=0.029 max=0.058 closed=n/a | settle=0.000 (n=100/100) | progress=0.016 net=0.001 yaw_rate_ratio=0.468
      back: err rms=0.049 initial=0.011 final=0.067 max=0.116 closed=n/a | settle=0.000 (n=100/100) | progress=-0.002 net=-0.001 yaw_rate_ratio=0.457
      diag: err rms=0.035 initial=0.011 final=0.035 max=0.081 closed=n/a | settle=0.000 (n=100/100) | progress=0.001 net=0.001 yaw_rate_ratio=0.352
  turn_l90: err rms=0.785 initial=1.570 final=0.024 max=1.574 closed=0.984 | settle=8.230 (n=100/100) | progress=0.173 net=0.132 yaw_rate_ratio=0.655
  turn_r90: err rms=0.792 initial=1.569 final=0.020 max=1.573 closed=0.988 | settle=8.340 (n=100/100) | progress=0.171 net=0.130 yaw_rate_ratio=0.644
  move_turn: err rms=1.093 initial=1.570 final=0.640 max=1.573 closed=0.589 | settle=n/a (n=0/100) | progress=0.077 net=0.078 yaw_rate_ratio=0.431
  move_turn_r: err rms=1.023 initial=1.567 final=0.367 max=1.576 closed=0.769 | settle=11.620 (n=2/100) | progress=0.100 net=0.100 yaw_rate_ratio=0.460

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.004 tripod=0.781 | closed=n/a progress=n/a net=0.001 | drift=0.031 dir_err=n/a
        fwd: n=100 surv=1.000 (fwd=1.00) | ratio=1.020 achieved=0.255 tripod=0.672 | closed=n/a progress=-0.003 net=0.002 | drift=0.254 dir_err=0.009
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=1.065 achieved=0.234 tripod=0.713 | closed=n/a progress=0.016 net=0.001 | drift=0.234 dir_err=0.020
       back: n=100 surv=1.000 (back=1.00) | ratio=1.005 achieved=0.221 tripod=0.694 | closed=n/a progress=-0.002 net=-0.001 | drift=0.222 dir_err=0.015
       diag: n=100 surv=1.000 (diag=1.00) | ratio=1.026 achieved=0.256 tripod=0.687 | closed=n/a progress=0.001 net=0.001 | drift=0.259 dir_err=0.058
       turn: n=200 surv=1.000 (turn_l90=1.00 turn_r90=1.00) | ratio=n/a achieved=0.000 tripod=0.787 | closed=0.986 progress=0.172 net=0.131 | drift=0.045 dir_err=n/a
  move_turn: n=200 surv=1.000 (move_turn=1.00 move_turn_r=1.00) | ratio=0.770 achieved=0.169 tripod=0.664 | closed=0.682 progress=0.087 net=0.088 | drift=0.181 dir_err=0.132

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.185 action_fft_freq=1.786 joint_speed_mean=0.365 rod_speed_mean=0.029 contact_switch_hz=1.410 foot_slip=0.077 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.018 settle_six_down_s=12.000 foot_disp_end=0.091 work_per_s=322.361
        fwd: action_jitter=0.122 action_fft_freq=1.077 joint_speed_mean=0.756 rod_speed_mean=0.029 contact_switch_hz=1.736 foot_slip=0.100 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=20.709 slip_per_m=1.174 work_per_m=1209.013 cam_work_per_m=455.482
       side: action_jitter=0.099 action_fft_freq=1.134 joint_speed_mean=0.388 rod_speed_mean=0.029 contact_switch_hz=1.340 foot_slip=0.079 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=17.343 slip_per_m=1.066 work_per_m=1122.523 cam_work_per_m=212.306
       back: action_jitter=0.130 action_fft_freq=1.158 joint_speed_mean=0.706 rod_speed_mean=0.029 contact_switch_hz=1.778 foot_slip=0.098 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=24.382 slip_per_m=1.355 work_per_m=1313.750 cam_work_per_m=434.192
       diag: action_jitter=0.111 action_fft_freq=1.187 joint_speed_mean=0.578 rod_speed_mean=0.029 contact_switch_hz=1.611 foot_slip=0.093 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=19.009 slip_per_m=1.084 work_per_m=1102.613 cam_work_per_m=232.067
       turn: action_jitter=0.137 action_fft_freq=1.177 joint_speed_mean=0.683 rod_speed_mean=0.029 contact_switch_hz=1.444 foot_slip=0.077 wz_active_frac=0.697 clock_restarts=0.000 swings_per_rad=32.791 slip_per_rad=1.848 work_per_rad=1999.630 cam_work_per_rad=631.324
  move_turn: action_jitter=0.114 action_fft_freq=1.077 joint_speed_mean=0.685 rod_speed_mean=0.028 contact_switch_hz=1.833 foot_slip=0.106 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=61.439 slip_per_rad=3.652 work_per_rad=3652.377 cam_work_per_rad=1299.823

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.235 steps_per_s=2.500 gait_hz=0.415 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=10.218 support_feet=3.004 taps_per_m=10.043 swing_apex=0.120 step_rate_hz=2.583 gait_hz=0.391 clock_hz=0.357 cadence_ratio=1.094 hz_speed_slope=0.093 offset_R=0.084
       side: steps_per_m=10.595 support_feet=3.107 taps_per_m=6.742 swing_apex=0.101 step_rate_hz=2.500 gait_hz=0.412 clock_hz=0.314 cadence_ratio=1.311 hz_speed_slope=1.989 offset_R=0.071
       back: steps_per_m=12.523 support_feet=3.032 taps_per_m=11.812 swing_apex=0.108 step_rate_hz=2.750 gait_hz=0.418 clock_hz=0.314 cadence_ratio=1.321 hz_speed_slope=-0.247 offset_R=0.036
       diag: steps_per_m=11.521 support_feet=2.990 taps_per_m=7.499 swing_apex=0.092 step_rate_hz=2.917 gait_hz=0.464 clock_hz=0.357 cadence_ratio=1.299 hz_speed_slope=-0.645 offset_R=0.138
       turn: steps_per_rad=19.379 support_feet=3.157 taps_per_rad=12.916 swing_apex=0.139 step_rate_hz=2.500 gait_hz=0.433 clock_hz=0.346 cadence_ratio=1.245 steps_per_rad_pre=13.495 steps_per_rad_post=5.756 offset_R=0.057
  move_turn: steps_per_rad=30.494 support_feet=3.048 taps_per_rad=31.437 swing_apex=0.111 step_rate_hz=2.667 gait_hz=0.366 clock_hz=0.500 cadence_ratio=0.732 hz_speed_slope=0.011 offset_R=0.055

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
