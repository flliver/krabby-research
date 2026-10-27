```
=== crab-hex gait eval ===
scenario   : suite__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_28000.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6372343755183996  p25=0.5627095091285079  p75=0.6900327776791269
tippy_tap_fraction median=0.3611111111111111
slip_ratio         median=0.1885370892083371
tracking_ratio     median=1.0103311088785762  (achieved/commanded vx, walking holds; n=700)
schedule_completion_rate=0.976
terminations={'schedule_complete': 976, 'fall': 24}
facing_error_rms   median=0.057225553645412174  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.23463228216866583 (n=736)
     stand: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7199212272180153 (n=100)
       fwd: cmd 0.25 -> achieved 0.254 m/s | tripod median=0.58167085543077 (n=100)
     lat_l: cmd 0.22 -> achieved 0.223 m/s | tripod median=0.6403691748730445 (n=100)
     lat_r: cmd 0.22 -> achieved 0.224 m/s | tripod median=0.7260575371254625 (n=100)
      back: cmd 0.22 -> achieved 0.226 m/s | tripod median=0.6616293861855314 (n=100)
      diag: cmd 0.25 -> achieved 0.260 m/s | tripod median=0.6493668912621566 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.012 m/s | tripod median=0.7420284431253843 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.010 m/s | tripod median=0.7388655301147472 (n=100)
  move_turn: cmd 0.22 -> achieved 0.186 m/s | tripod median=0.43555820502434206 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.170 m/s | tripod median=0.5620885746662339 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.011 initial=0.010 final=0.013 max=0.024 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.001 yaw_rate_ratio=n/a
     stand: err rms=0.024 initial=0.013 final=0.022 max=0.054 closed=n/a | settle=0.000 (n=100/100) | progress=0.015 net=0.000 yaw_rate_ratio=0.124
       fwd: err rms=0.045 initial=0.013 final=0.046 max=0.114 closed=n/a | settle=0.000 (n=98/100) | progress=-0.008 net=0.004 yaw_rate_ratio=0.738
     lat_l: err rms=0.021 initial=0.014 final=0.023 max=0.049 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.002 yaw_rate_ratio=n/a
     lat_r: err rms=0.024 initial=0.011 final=0.024 max=0.058 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a
      back: err rms=0.081 initial=0.015 final=0.098 max=0.160 closed=n/a | settle=8.760 (n=95/100) | progress=-0.004 net=-0.005 yaw_rate_ratio=0.726
      diag: err rms=0.032 initial=0.015 final=0.033 max=0.080 closed=n/a | settle=0.000 (n=99/100) | progress=-0.002 net=0.001 yaw_rate_ratio=0.420
  turn_l90: err rms=0.797 initial=1.561 final=0.013 max=1.567 closed=0.992 | settle=8.590 (n=100/100) | progress=0.166 net=0.129 yaw_rate_ratio=0.639
  turn_r90: err rms=0.812 initial=1.579 final=0.032 max=1.584 closed=0.979 | settle=8.700 (n=100/100) | progress=0.165 net=0.129 yaw_rate_ratio=0.638
  move_turn: err rms=1.206 initial=1.557 final=0.938 max=1.574 closed=0.388 | settle=n/a (n=0/100) | progress=0.051 net=0.052 yaw_rate_ratio=0.449
  move_turn_r: err rms=1.032 initial=1.582 final=0.423 max=1.587 closed=0.732 | settle=11.600 (n=1/100) | progress=0.096 net=0.097 yaw_rate_ratio=0.469

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=0.990 (stand=0.99) | ratio=n/a achieved=0.003 tripod=0.720 | closed=n/a progress=0.015 net=0.000 | drift=0.033 dir_err=n/a
        fwd: n=100 surv=0.990 (fwd=0.99) | ratio=1.018 achieved=0.254 tripod=0.582 | closed=n/a progress=-0.008 net=0.004 | drift=0.255 dir_err=0.021
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=1.015 achieved=0.223 tripod=0.669 | closed=n/a progress=n/a net=0.001 | drift=0.225 dir_err=0.019
       back: n=100 surv=1.000 (back=1.00) | ratio=1.025 achieved=0.226 tripod=0.662 | closed=n/a progress=-0.004 net=-0.005 | drift=0.226 dir_err=0.016
       diag: n=100 surv=0.990 (diag=0.99) | ratio=1.039 achieved=0.260 tripod=0.649 | closed=n/a progress=-0.002 net=0.001 | drift=0.261 dir_err=0.034
       turn: n=200 surv=1.000 (turn_l90=1.00 turn_r90=1.00) | ratio=n/a achieved=-0.000 tripod=0.740 | closed=0.986 progress=0.166 net=0.129 | drift=0.050 dir_err=n/a
  move_turn: n=200 surv=0.895 (move_turn=0.80 move_turn_r=0.99) | ratio=0.811 achieved=0.178 tripod=0.492 | closed=0.493 progress=0.066 net=0.066 | drift=0.192 dir_err=0.112

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.108 action_fft_freq=1.238 joint_speed_mean=0.363 rod_speed_mean=0.028 contact_switch_hz=1.125 foot_slip=0.089 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.040 settle_six_down_s=12.000 foot_disp_end=0.102 work_per_s=433.084
        fwd: action_jitter=0.081 action_fft_freq=0.904 joint_speed_mean=0.753 rod_speed_mean=0.029 contact_switch_hz=1.535 foot_slip=0.117 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=18.505 slip_per_m=1.509 work_per_m=1609.752 cam_work_per_m=680.178
       side: action_jitter=0.066 action_fft_freq=0.945 joint_speed_mean=0.385 rod_speed_mean=0.028 contact_switch_hz=1.292 foot_slip=0.101 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=17.307 slip_per_m=1.486 work_per_m=1639.119 cam_work_per_m=313.254
       back: action_jitter=0.088 action_fft_freq=0.946 joint_speed_mean=0.713 rod_speed_mean=0.029 contact_switch_hz=1.549 foot_slip=0.113 wz_active_frac=0.044 clock_restarts=0.000 swings_per_m=21.076 slip_per_m=1.667 work_per_m=1716.803 cam_work_per_m=638.069
       diag: action_jitter=0.083 action_fft_freq=1.004 joint_speed_mean=0.578 rod_speed_mean=0.029 contact_switch_hz=1.583 foot_slip=0.119 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=18.692 slip_per_m=1.487 work_per_m=1438.967 cam_work_per_m=316.881
       turn: action_jitter=0.088 action_fft_freq=0.929 joint_speed_mean=0.691 rod_speed_mean=0.029 contact_switch_hz=1.194 foot_slip=0.095 wz_active_frac=0.723 clock_restarts=0.000 swings_per_rad=28.382 slip_per_rad=2.449 work_per_rad=2694.778 cam_work_per_rad=894.060
  move_turn: action_jitter=0.078 action_fft_freq=0.917 joint_speed_mean=0.669 rod_speed_mean=0.027 contact_switch_hz=1.653 foot_slip=0.126 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=69.958 slip_per_rad=6.059 work_per_rad=6406.641 cam_work_per_rad=2967.703

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.426 steps_per_s=2.167 gait_hz=0.348 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=11.055 support_feet=3.227 taps_per_m=7.653 swing_apex=0.114 step_rate_hz=2.792 gait_hz=0.385 clock_hz=0.357 cadence_ratio=1.077 hz_speed_slope=-0.444 offset_R=0.122
       side: steps_per_m=11.136 support_feet=3.287 taps_per_m=6.082 swing_apex=0.095 step_rate_hz=2.500 gait_hz=0.381 clock_hz=0.314 cadence_ratio=1.214 hz_speed_slope=0.084 offset_R=0.048
       back: steps_per_m=12.366 support_feet=3.253 taps_per_m=8.225 swing_apex=0.104 step_rate_hz=2.750 gait_hz=0.403 clock_hz=0.317 cadence_ratio=1.262 hz_speed_slope=0.509 offset_R=0.122
       diag: steps_per_m=11.506 support_feet=3.178 taps_per_m=6.948 swing_apex=0.093 step_rate_hz=3.000 gait_hz=0.439 clock_hz=0.357 cadence_ratio=1.230 hz_speed_slope=-0.395 offset_R=0.189
       turn: steps_per_rad=19.869 support_feet=3.310 taps_per_rad=8.267 swing_apex=0.136 step_rate_hz=2.583 gait_hz=0.403 clock_hz=0.359 cadence_ratio=1.127 steps_per_rad_pre=15.067 steps_per_rad_post=4.725 offset_R=0.021
  move_turn: steps_per_rad=40.040 support_feet=3.195 taps_per_rad=30.365 swing_apex=0.090 step_rate_hz=2.750 gait_hz=0.360 clock_hz=0.500 cadence_ratio=0.720 hz_speed_slope=-0.199 offset_R=0.038

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
