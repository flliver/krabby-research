```
=== crab-hex gait eval ===
scenario   : suite__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-10-21_15-52-28_seed19_X01_s1_A15pB/model_5999.pt
episodes   : 1000  unscored: 1
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6064233227942278  p25=0.40740816175091843  p75=0.6922451042348375
tippy_tap_fraction median=0.5358983799705449
slip_ratio         median=0.30859308439605226
tracking_ratio     median=0.8608104150469658  (achieved/commanded vx, walking holds; n=698)
schedule_completion_rate=0.812
terminations={'schedule_complete': 812, 'fall': 188}
facing_error_rms   median=0.07406010663175835  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved -0.002 m/s | tripod median=0.09590215239456963 (n=881)
     stand: cmd 0.00 -> achieved -0.001 m/s | tripod median=0.19442710576186956 (n=98)
       fwd: cmd 0.25 -> achieved 0.220 m/s | tripod median=0.6769276292263715 (n=100)
     lat_l: cmd 0.22 -> achieved 0.191 m/s | tripod median=0.7897910168122995 (n=100)
     lat_r: cmd 0.22 -> achieved 0.187 m/s | tripod median=0.750896000998093 (n=100)
      back: cmd 0.22 -> achieved 0.199 m/s | tripod median=0.5624593233674542 (n=100)
      diag: cmd 0.25 -> achieved 0.217 m/s | tripod median=0.6004257959807733 (n=100)
  turn_l90: cmd 0.00 -> achieved 0.029 m/s | tripod median=0.7358188245749281 (n=100)
  turn_r90: cmd 0.00 -> achieved -0.018 m/s | tripod median=0.6974218853320431 (n=100)
  move_turn: cmd 0.22 -> achieved 0.159 m/s | tripod median=0.3080214572398242 (n=98)
  move_turn_r: cmd 0.22 -> achieved 0.077 m/s | tripod median=0.3020027429727822 (n=99)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.007 initial=0.006 final=0.006 max=0.013 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.001 yaw_rate_ratio=n/a
     stand: err rms=0.011 initial=0.005 final=0.014 max=0.022 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.000 yaw_rate_ratio=n/a
       fwd: err rms=0.089 initial=0.006 final=0.135 max=0.193 closed=n/a | settle=11.150 (n=64/100) | progress=-0.002 net=0.008 yaw_rate_ratio=0.419
     lat_l: err rms=0.023 initial=0.006 final=0.036 max=0.055 closed=n/a | settle=0.000 (n=100/100) | progress=-0.025 net=0.001 yaw_rate_ratio=0.499
     lat_r: err rms=0.032 initial=0.008 final=0.048 max=0.064 closed=n/a | settle=0.000 (n=100/100) | progress=-0.007 net=0.001 yaw_rate_ratio=0.279
      back: err rms=0.076 initial=0.008 final=0.114 max=0.146 closed=n/a | settle=0.000 (n=87/100) | progress=-0.003 net=-0.006 yaw_rate_ratio=0.331
      diag: err rms=0.032 initial=0.006 final=0.025 max=0.075 closed=n/a | settle=0.000 (n=100/100) | progress=-0.002 net=0.002 yaw_rate_ratio=0.483
  turn_l90: err rms=1.427 initial=1.571 final=1.410 max=1.571 closed=0.103 | settle=n/a (n=0/100) | progress=0.013 net=0.013 yaw_rate_ratio=0.141
  turn_r90: err rms=1.439 initial=1.571 final=1.453 max=1.572 closed=0.075 | settle=n/a (n=0/100) | progress=0.009 net=0.010 yaw_rate_ratio=0.127
  move_turn: err rms=1.459 initial=1.570 final=1.426 max=1.573 closed=0.091 | settle=n/a (n=0/100) | progress=0.036 net=0.040 yaw_rate_ratio=0.462
  move_turn_r: err rms=1.507 initial=1.572 final=1.529 max=1.675 closed=0.025 | settle=n/a (n=0/100) | progress=0.005 net=0.007 yaw_rate_ratio=0.555

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=-0.001 tripod=0.194 | closed=n/a progress=n/a net=-0.000 | drift=0.031 dir_err=n/a
        fwd: n=100 surv=1.000 (fwd=1.00) | ratio=0.880 achieved=0.220 tripod=0.677 | closed=n/a progress=-0.002 net=0.008 | drift=0.221 dir_err=0.014
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=0.862 achieved=0.190 tripod=0.772 | closed=n/a progress=-0.013 net=0.001 | drift=0.193 dir_err=0.019
       back: n=100 surv=1.000 (back=1.00) | ratio=0.904 achieved=0.199 tripod=0.562 | closed=n/a progress=-0.003 net=-0.006 | drift=0.200 dir_err=0.024
       diag: n=100 surv=1.000 (diag=1.00) | ratio=0.867 achieved=0.217 tripod=0.600 | closed=n/a progress=-0.002 net=0.002 | drift=0.219 dir_err=0.013
       turn: n=200 surv=1.000 (turn_l90=1.00 turn_r90=1.00) | ratio=n/a achieved=0.007 tripod=0.716 | closed=0.087 progress=0.011 net=0.011 | drift=0.043 dir_err=n/a
  move_turn: n=200 surv=0.060 (move_turn=0.00 move_turn_r=0.12) | ratio=0.500 achieved=0.110 tripod=0.302 | closed=0.067 progress=0.019 net=0.022 | drift=0.155 dir_err=0.085

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.296 action_fft_freq=3.752 joint_speed_mean=0.254 rod_speed_mean=0.029 contact_switch_hz=2.799 foot_slip=0.052 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.008 settle_six_down_s=12.000 foot_disp_end=0.058 work_per_s=224.415
        fwd: action_jitter=0.195 action_fft_freq=1.521 joint_speed_mean=0.680 rod_speed_mean=0.028 contact_switch_hz=2.000 foot_slip=0.091 wz_active_frac=0.148 clock_restarts=0.000 swings_per_m=27.528 slip_per_m=1.216 work_per_m=1100.862 cam_work_per_m=408.925
       side: action_jitter=0.136 action_fft_freq=1.224 joint_speed_mean=0.309 rod_speed_mean=0.028 contact_switch_hz=1.306 foot_slip=0.091 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=20.933 slip_per_m=1.481 work_per_m=1389.003 cam_work_per_m=328.505
       back: action_jitter=0.214 action_fft_freq=1.750 joint_speed_mean=0.626 rod_speed_mean=0.029 contact_switch_hz=2.299 foot_slip=0.084 wz_active_frac=0.037 clock_restarts=0.000 swings_per_m=34.749 slip_per_m=1.247 work_per_m=1151.734 cam_work_per_m=382.501
       diag: action_jitter=0.181 action_fft_freq=1.421 joint_speed_mean=0.564 rod_speed_mean=0.028 contact_switch_hz=2.188 foot_slip=0.095 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=30.859 slip_per_m=1.308 work_per_m=1082.488 cam_work_per_m=307.685
       turn: action_jitter=0.249 action_fft_freq=2.326 joint_speed_mean=0.369 rod_speed_mean=0.030 contact_switch_hz=2.347 foot_slip=0.057 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=610.046 slip_per_rad=15.948 work_per_rad=18488.801 cam_work_per_rad=2406.319
  move_turn: action_jitter=0.161 action_fft_freq=1.588 joint_speed_mean=0.721 rod_speed_mean=0.029 contact_switch_hz=2.432 foot_slip=0.119 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=250.899 slip_per_rad=10.892 work_per_rad=8636.386 cam_work_per_rad=4992.148

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.405 steps_per_s=3.167 gait_hz=1.205 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=11.675 support_feet=2.923 taps_per_m=15.827 swing_apex=0.100 step_rate_hz=2.583 gait_hz=0.366 clock_hz=0.365 cadence_ratio=0.992 hz_speed_slope=0.129 offset_R=0.996
       side: steps_per_m=10.591 support_feet=3.058 taps_per_m=9.940 swing_apex=0.112 step_rate_hz=2.000 gait_hz=0.311 clock_hz=0.314 cadence_ratio=0.990 hz_speed_slope=0.003 offset_R=0.997
       back: steps_per_m=14.384 support_feet=2.931 taps_per_m=20.467 swing_apex=0.083 step_rate_hz=2.875 gait_hz=0.317 clock_hz=0.316 cadence_ratio=1.006 hz_speed_slope=0.272 offset_R=0.997
       diag: steps_per_m=11.898 support_feet=2.981 taps_per_m=18.594 swing_apex=0.097 step_rate_hz=2.583 gait_hz=0.360 clock_hz=0.357 cadence_ratio=1.008 hz_speed_slope=-0.034 offset_R=0.998
       turn: steps_per_rad=264.130 support_feet=3.151 taps_per_rad=352.625 swing_apex=0.084 step_rate_hz=3.000 gait_hz=0.500 clock_hz=0.500 cadence_ratio=1.001 steps_per_rad_pre=264.130 steps_per_rad_post=n/a offset_R=0.999
  move_turn: steps_per_rad=109.821 support_feet=2.882 taps_per_rad=126.985 swing_apex=0.055 step_rate_hz=3.286 gait_hz=0.415 clock_hz=0.500 cadence_ratio=0.830 hz_speed_slope=1.374 offset_R=0.966

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
