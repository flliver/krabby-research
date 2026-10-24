```
=== crab-hex gait eval ===
scenario   : suite__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_28000.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6865690075462371  p25=0.6449506579081462  p75=0.7355981317360123
tippy_tap_fraction median=0.3680340557275542
slip_ratio         median=0.19644488414320566
tracking_ratio     median=1.0131106785808837  (achieved/commanded vx, walking holds; n=700)
schedule_completion_rate=1.0
terminations={'schedule_complete': 1000}
facing_error_rms   median=0.06142493398062131  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.5977189429156042 (n=931)
     stand: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7391506371883334 (n=100)
       fwd: cmd 0.25 -> achieved 0.253 m/s | tripod median=0.6690988709101265 (n=100)
     lat_l: cmd 0.22 -> achieved 0.231 m/s | tripod median=0.6600357472286947 (n=100)
     lat_r: cmd 0.22 -> achieved 0.231 m/s | tripod median=0.7474358398821273 (n=100)
      back: cmd 0.22 -> achieved 0.221 m/s | tripod median=0.7015119666311667 (n=100)
      diag: cmd 0.25 -> achieved 0.257 m/s | tripod median=0.6854559641586296 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.008 m/s | tripod median=0.7763570404885938 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.007 m/s | tripod median=0.7744429739054138 (n=100)
  move_turn: cmd 0.22 -> achieved 0.170 m/s | tripod median=0.5731723875631791 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.169 m/s | tripod median=0.6744685628338568 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.012 initial=0.008 final=0.015 max=0.028 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=-0.000 yaw_rate_ratio=n/a
     stand: err rms=0.029 initial=0.015 final=0.034 max=0.062 closed=n/a | settle=0.000 (n=100/100) | progress=0.000 net=-0.000 yaw_rate_ratio=0.328
       fwd: err rms=0.043 initial=0.014 final=0.033 max=0.107 closed=n/a | settle=0.000 (n=100/100) | progress=-0.005 net=0.002 yaw_rate_ratio=0.691
     lat_l: err rms=0.022 initial=0.016 final=0.023 max=0.048 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
     lat_r: err rms=0.028 initial=0.014 final=0.023 max=0.061 closed=n/a | settle=0.000 (n=100/100) | progress=-0.002 net=0.000 yaw_rate_ratio=0.182
      back: err rms=0.084 initial=0.017 final=0.089 max=0.152 closed=n/a | settle=6.200 (n=99/100) | progress=-0.000 net=-0.004 yaw_rate_ratio=0.639
      diag: err rms=0.039 initial=0.012 final=0.051 max=0.089 closed=n/a | settle=0.000 (n=100/100) | progress=-0.001 net=0.004 yaw_rate_ratio=0.447
  turn_l90: err rms=0.785 initial=1.559 final=0.019 max=1.567 closed=0.988 | settle=8.360 (n=100/100) | progress=0.170 net=0.131 yaw_rate_ratio=0.655
  turn_r90: err rms=0.818 initial=1.583 final=0.021 max=1.592 closed=0.987 | settle=8.580 (n=100/100) | progress=0.168 net=0.131 yaw_rate_ratio=0.647
  move_turn: err rms=1.141 initial=1.559 final=0.770 max=1.572 closed=0.510 | settle=n/a (n=0/100) | progress=0.066 net=0.066 yaw_rate_ratio=0.429
  move_turn_r: err rms=1.004 initial=1.583 final=0.349 max=1.588 closed=0.779 | settle=n/a (n=0/100) | progress=0.102 net=0.102 yaw_rate_ratio=0.466

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.003 tripod=0.739 | closed=n/a progress=0.000 net=-0.000 | drift=0.030 dir_err=n/a
        fwd: n=100 surv=1.000 (fwd=1.00) | ratio=1.010 achieved=0.253 tripod=0.669 | closed=n/a progress=-0.005 net=0.002 | drift=0.252 dir_err=0.018
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=1.048 achieved=0.231 tripod=0.705 | closed=n/a progress=-0.002 net=0.001 | drift=0.232 dir_err=0.020
       back: n=100 surv=1.000 (back=1.00) | ratio=1.004 achieved=0.221 tripod=0.702 | closed=n/a progress=-0.000 net=-0.004 | drift=0.221 dir_err=0.022
       diag: n=100 surv=1.000 (diag=1.00) | ratio=1.026 achieved=0.257 tripod=0.685 | closed=n/a progress=-0.001 net=0.004 | drift=0.260 dir_err=0.052
       turn: n=200 surv=1.000 (turn_l90=1.00 turn_r90=1.00) | ratio=n/a achieved=-0.000 tripod=0.776 | closed=0.987 progress=0.169 net=0.131 | drift=0.046 dir_err=n/a
  move_turn: n=200 surv=1.000 (move_turn=1.00 move_turn_r=1.00) | ratio=0.770 achieved=0.169 tripod=0.616 | closed=0.634 progress=0.082 net=0.083 | drift=0.183 dir_err=0.125

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.124 action_fft_freq=1.385 joint_speed_mean=0.359 rod_speed_mean=0.028 contact_switch_hz=1.222 foot_slip=0.083 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.025 settle_six_down_s=12.000 foot_disp_end=0.113 work_per_s=371.046
        fwd: action_jitter=0.092 action_fft_freq=0.963 joint_speed_mean=0.755 rod_speed_mean=0.029 contact_switch_hz=1.583 foot_slip=0.106 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=19.166 slip_per_m=1.293 work_per_m=1361.513 cam_work_per_m=543.319
       side: action_jitter=0.073 action_fft_freq=1.006 joint_speed_mean=0.383 rod_speed_mean=0.028 contact_switch_hz=1.208 foot_slip=0.085 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=15.801 slip_per_m=1.175 work_per_m=1292.863 cam_work_per_m=233.780
       back: action_jitter=0.100 action_fft_freq=1.016 joint_speed_mean=0.704 rod_speed_mean=0.029 contact_switch_hz=1.618 foot_slip=0.103 wz_active_frac=0.028 clock_restarts=0.000 swings_per_m=22.227 slip_per_m=1.466 work_per_m=1452.455 cam_work_per_m=487.270
       diag: action_jitter=0.086 action_fft_freq=1.071 joint_speed_mean=0.563 rod_speed_mean=0.029 contact_switch_hz=1.521 foot_slip=0.100 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=18.085 slip_per_m=1.183 work_per_m=1202.502 cam_work_per_m=244.967
       turn: action_jitter=0.097 action_fft_freq=0.997 joint_speed_mean=0.687 rod_speed_mean=0.029 contact_switch_hz=1.264 foot_slip=0.082 wz_active_frac=0.712 clock_restarts=0.000 swings_per_rad=29.158 slip_per_rad=2.007 work_per_rad=2227.318 cam_work_per_rad=761.720
  move_turn: action_jitter=0.084 action_fft_freq=0.968 joint_speed_mean=0.679 rod_speed_mean=0.028 contact_switch_hz=1.639 foot_slip=0.112 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=59.316 slip_per_rad=4.189 work_per_rad=4150.893 cam_work_per_rad=1620.293

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.337 steps_per_s=2.250 gait_hz=0.369 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=10.426 support_feet=3.085 taps_per_m=8.534 swing_apex=0.120 step_rate_hz=2.583 gait_hz=0.391 clock_hz=0.357 cadence_ratio=1.094 hz_speed_slope=-0.489 offset_R=0.099
       side: steps_per_m=10.669 support_feet=3.154 taps_per_m=5.190 swing_apex=0.101 step_rate_hz=2.458 gait_hz=0.388 clock_hz=0.314 cadence_ratio=1.233 hz_speed_slope=1.895 offset_R=0.089
       back: steps_per_m=12.288 support_feet=3.112 taps_per_m=9.695 swing_apex=0.110 step_rate_hz=2.667 gait_hz=0.409 clock_hz=0.316 cadence_ratio=1.292 hz_speed_slope=0.252 offset_R=0.082
       diag: steps_per_m=11.602 support_feet=3.041 taps_per_m=6.319 swing_apex=0.090 step_rate_hz=3.000 gait_hz=0.446 clock_hz=0.357 cadence_ratio=1.248 hz_speed_slope=0.892 offset_R=0.065
       turn: steps_per_rad=19.555 support_feet=3.212 taps_per_rad=9.277 swing_apex=0.138 step_rate_hz=2.583 gait_hz=0.421 clock_hz=0.354 cadence_ratio=1.205 steps_per_rad_pre=13.917 steps_per_rad_post=5.706 offset_R=0.014
  move_turn: steps_per_rad=31.421 support_feet=3.083 taps_per_rad=26.544 swing_apex=0.107 step_rate_hz=2.667 gait_hz=0.366 clock_hz=0.500 cadence_ratio=0.732 hz_speed_slope=-0.213 offset_R=0.063

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
