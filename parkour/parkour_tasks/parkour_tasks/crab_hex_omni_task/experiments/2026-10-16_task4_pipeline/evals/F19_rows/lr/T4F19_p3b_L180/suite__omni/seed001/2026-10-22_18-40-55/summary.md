```
=== crab-hex gait eval ===
scenario   : suite__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6318042951788675  p25=0.5492219898725356  p75=0.6865548996630737
tippy_tap_fraction median=0.4189189189189189
slip_ratio         median=0.20584803046071126
tracking_ratio     median=1.0256897409592005  (achieved/commanded vx, walking holds; n=700)
schedule_completion_rate=0.969
terminations={'schedule_complete': 969, 'fall': 31}
facing_error_rms   median=0.0479417791676462  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.48699134519320236 (n=914)
     stand: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7085103122958355 (n=100)
       fwd: cmd 0.25 -> achieved 0.261 m/s | tripod median=0.567106631791449 (n=100)
     lat_l: cmd 0.22 -> achieved 0.227 m/s | tripod median=0.6452503601791248 (n=100)
     lat_r: cmd 0.22 -> achieved 0.225 m/s | tripod median=0.745642153210589 (n=100)
      back: cmd 0.22 -> achieved 0.228 m/s | tripod median=0.659963134374262 (n=100)
      diag: cmd 0.25 -> achieved 0.261 m/s | tripod median=0.623916283439458 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.013 m/s | tripod median=0.711187561327125 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.010 m/s | tripod median=0.7009074774331849 (n=100)
  move_turn: cmd 0.22 -> achieved 0.182 m/s | tripod median=0.44627115756181723 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.174 m/s | tripod median=0.5024292938449166 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.011 initial=0.008 final=0.012 max=0.024 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.002 yaw_rate_ratio=n/a
     stand: err rms=0.021 initial=0.013 final=0.023 max=0.050 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.037 initial=0.012 final=0.035 max=0.090 closed=n/a | settle=0.000 (n=99/100) | progress=-0.004 net=0.001 yaw_rate_ratio=0.553
     lat_l: err rms=0.019 initial=0.012 final=0.016 max=0.047 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
     lat_r: err rms=0.024 initial=0.012 final=0.020 max=0.054 closed=n/a | settle=0.000 (n=100/100) | progress=0.007 net=0.001 yaw_rate_ratio=0.155
      back: err rms=0.065 initial=0.013 final=0.071 max=0.133 closed=n/a | settle=0.000 (n=100/100) | progress=-0.003 net=-0.003 yaw_rate_ratio=0.735
      diag: err rms=0.030 initial=0.013 final=0.027 max=0.070 closed=n/a | settle=0.000 (n=99/100) | progress=-0.004 net=0.002 yaw_rate_ratio=0.620
  turn_l90: err rms=0.797 initial=1.568 final=0.014 max=1.578 closed=0.991 | settle=8.560 (n=100/100) | progress=0.167 net=0.131 yaw_rate_ratio=0.626
  turn_r90: err rms=0.797 initial=1.570 final=0.032 max=1.579 closed=0.979 | settle=8.690 (n=100/100) | progress=0.165 net=0.128 yaw_rate_ratio=0.628
  move_turn: err rms=1.227 initial=1.570 final=0.965 max=1.582 closed=0.385 | settle=n/a (n=0/100) | progress=0.050 net=0.051 yaw_rate_ratio=0.450
  move_turn_r: err rms=1.032 initial=1.574 final=0.537 max=1.579 closed=0.658 | settle=n/a (n=0/100) | progress=0.086 net=0.087 yaw_rate_ratio=0.472

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.003 tripod=0.709 | closed=n/a progress=n/a net=0.001 | drift=0.032 dir_err=n/a
        fwd: n=100 surv=0.970 (fwd=0.97) | ratio=1.045 achieved=0.261 tripod=0.567 | closed=n/a progress=-0.004 net=0.001 | drift=0.261 dir_err=0.010
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=1.029 achieved=0.226 tripod=0.677 | closed=n/a progress=0.007 net=0.001 | drift=0.228 dir_err=0.019
       back: n=100 surv=1.000 (back=1.00) | ratio=1.037 achieved=0.228 tripod=0.660 | closed=n/a progress=-0.003 net=-0.003 | drift=0.229 dir_err=0.014
       diag: n=100 surv=0.960 (diag=0.96) | ratio=1.043 achieved=0.261 tripod=0.624 | closed=n/a progress=-0.004 net=0.002 | drift=0.262 dir_err=0.048
       turn: n=200 surv=1.000 (turn_l90=1.00 turn_r90=1.00) | ratio=n/a achieved=-0.001 tripod=0.705 | closed=0.987 progress=0.166 net=0.130 | drift=0.048 dir_err=n/a
  move_turn: n=200 surv=0.880 (move_turn=0.89 move_turn_r=0.87) | ratio=0.807 achieved=0.178 tripod=0.482 | closed=0.500 progress=0.066 net=0.067 | drift=0.194 dir_err=0.096

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.146 action_fft_freq=1.476 joint_speed_mean=0.371 rod_speed_mean=0.029 contact_switch_hz=1.417 foot_slip=0.087 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.033 settle_six_down_s=12.000 foot_disp_end=0.113 work_per_s=404.543
        fwd: action_jitter=0.106 action_fft_freq=0.982 joint_speed_mean=0.755 rod_speed_mean=0.029 contact_switch_hz=1.667 foot_slip=0.116 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=19.549 slip_per_m=1.414 work_per_m=1570.541 cam_work_per_m=646.137
       side: action_jitter=0.084 action_fft_freq=1.022 joint_speed_mean=0.387 rod_speed_mean=0.029 contact_switch_hz=1.278 foot_slip=0.097 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=17.812 slip_per_m=1.410 work_per_m=1593.632 cam_work_per_m=281.797
       back: action_jitter=0.110 action_fft_freq=1.009 joint_speed_mean=0.716 rod_speed_mean=0.029 contact_switch_hz=1.632 foot_slip=0.111 wz_active_frac=0.002 clock_restarts=0.000 swings_per_m=21.895 slip_per_m=1.612 work_per_m=1657.558 cam_work_per_m=591.218
       diag: action_jitter=0.100 action_fft_freq=1.090 joint_speed_mean=0.578 rod_speed_mean=0.029 contact_switch_hz=1.681 foot_slip=0.125 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=19.708 slip_per_m=1.541 work_per_m=1418.349 cam_work_per_m=320.558
       turn: action_jitter=0.110 action_fft_freq=1.014 joint_speed_mean=0.691 rod_speed_mean=0.029 contact_switch_hz=1.389 foot_slip=0.100 wz_active_frac=0.725 clock_restarts=0.000 swings_per_rad=32.191 slip_per_rad=2.547 work_per_rad=2701.082 cam_work_per_rad=831.091
  move_turn: action_jitter=0.101 action_fft_freq=1.007 joint_speed_mean=0.670 rod_speed_mean=0.028 contact_switch_hz=1.694 foot_slip=0.126 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=74.411 slip_per_rad=6.060 work_per_rad=6099.989 cam_work_per_rad=2764.411

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.380 steps_per_s=2.500 gait_hz=0.391 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=10.518 support_feet=3.184 taps_per_m=8.808 swing_apex=0.114 step_rate_hz=2.750 gait_hz=0.385 clock_hz=0.357 cadence_ratio=1.076 hz_speed_slope=-0.412 offset_R=0.124
       side: steps_per_m=11.010 support_feet=3.278 taps_per_m=6.239 swing_apex=0.098 step_rate_hz=2.417 gait_hz=0.391 clock_hz=0.314 cadence_ratio=1.243 hz_speed_slope=2.183 offset_R=0.020
       back: steps_per_m=12.031 support_feet=3.242 taps_per_m=9.733 swing_apex=0.108 step_rate_hz=2.708 gait_hz=0.409 clock_hz=0.314 cadence_ratio=1.282 hz_speed_slope=-0.296 offset_R=0.188
       diag: steps_per_m=11.700 support_feet=3.184 taps_per_m=7.941 swing_apex=0.089 step_rate_hz=3.000 gait_hz=0.452 clock_hz=0.357 cadence_ratio=1.265 hz_speed_slope=-0.313 offset_R=0.083
       turn: steps_per_rad=19.674 support_feet=3.303 taps_per_rad=12.202 swing_apex=0.133 step_rate_hz=2.583 gait_hz=0.403 clock_hz=0.359 cadence_ratio=1.122 steps_per_rad_pre=14.241 steps_per_rad_post=5.183 offset_R=0.045
  move_turn: steps_per_rad=40.124 support_feet=3.148 taps_per_rad=33.840 swing_apex=0.093 step_rate_hz=2.750 gait_hz=0.366 clock_hz=0.500 cadence_ratio=0.732 hz_speed_slope=-0.404 offset_R=0.098

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
