```
=== crab-hex gait eval ===
scenario   : suite__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6668413904058237  p25=0.6165779177431308  p75=0.7255642087826155
tippy_tap_fraction median=0.4100462379150904
slip_ratio         median=0.20776466819156345
tracking_ratio     median=1.0338113446986519  (achieved/commanded vx, walking holds; n=700)
schedule_completion_rate=0.999
terminations={'schedule_complete': 999, 'fall': 1}
facing_error_rms   median=0.04229462407966614  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.6226347744228613 (n=966)
     stand: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7375946455882465 (n=100)
       fwd: cmd 0.25 -> achieved 0.260 m/s | tripod median=0.6396020495746296 (n=100)
     lat_l: cmd 0.22 -> achieved 0.234 m/s | tripod median=0.6546768654611974 (n=100)
     lat_r: cmd 0.22 -> achieved 0.230 m/s | tripod median=0.7611332761490461 (n=100)
      back: cmd 0.22 -> achieved 0.226 m/s | tripod median=0.6938618454190837 (n=100)
      diag: cmd 0.25 -> achieved 0.261 m/s | tripod median=0.6442990812076125 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.010 m/s | tripod median=0.7470400893826168 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.010 m/s | tripod median=0.7373052721088798 (n=100)
  move_turn: cmd 0.22 -> achieved 0.175 m/s | tripod median=0.537975324017306 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.173 m/s | tripod median=0.6054537008176111 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.011 initial=0.008 final=0.012 max=0.025 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.002 yaw_rate_ratio=n/a
     stand: err rms=0.021 initial=0.013 final=0.021 max=0.051 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.035 initial=0.012 final=0.036 max=0.082 closed=n/a | settle=0.000 (n=99/100) | progress=-0.002 net=0.001 yaw_rate_ratio=0.600
     lat_l: err rms=0.018 initial=0.013 final=0.016 max=0.046 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
     lat_r: err rms=0.026 initial=0.011 final=0.027 max=0.056 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
      back: err rms=0.055 initial=0.011 final=0.060 max=0.117 closed=n/a | settle=0.000 (n=100/100) | progress=-0.001 net=0.000 yaw_rate_ratio=0.557
      diag: err rms=0.031 initial=0.013 final=0.029 max=0.076 closed=n/a | settle=0.000 (n=100/100) | progress=0.001 net=0.002 yaw_rate_ratio=0.343
  turn_l90: err rms=0.794 initial=1.572 final=0.018 max=1.577 closed=0.989 | settle=8.430 (n=100/100) | progress=0.170 net=0.132 yaw_rate_ratio=0.631
  turn_r90: err rms=0.789 initial=1.569 final=0.024 max=1.578 closed=0.985 | settle=8.500 (n=100/100) | progress=0.168 net=0.130 yaw_rate_ratio=0.642
  move_turn: err rms=1.160 initial=1.571 final=0.847 max=1.578 closed=0.461 | settle=n/a (n=0/100) | progress=0.060 net=0.061 yaw_rate_ratio=0.440
  move_turn_r: err rms=1.010 initial=1.570 final=0.420 max=1.578 closed=0.733 | settle=n/a (n=0/100) | progress=0.096 net=0.097 yaw_rate_ratio=0.479

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.003 tripod=0.738 | closed=n/a progress=n/a net=0.001 | drift=0.031 dir_err=n/a
        fwd: n=100 surv=1.000 (fwd=1.00) | ratio=1.042 achieved=0.260 tripod=0.640 | closed=n/a progress=-0.002 net=0.001 | drift=0.260 dir_err=0.011
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=1.056 achieved=0.232 tripod=0.707 | closed=n/a progress=n/a net=0.001 | drift=0.233 dir_err=0.020
       back: n=100 surv=1.000 (back=1.00) | ratio=1.028 achieved=0.226 tripod=0.694 | closed=n/a progress=-0.001 net=0.000 | drift=0.226 dir_err=0.007
       diag: n=100 surv=1.000 (diag=1.00) | ratio=1.042 achieved=0.261 tripod=0.644 | closed=n/a progress=0.001 net=0.002 | drift=0.263 dir_err=0.045
       turn: n=200 surv=1.000 (turn_l90=1.00 turn_r90=1.00) | ratio=n/a achieved=-0.002 tripod=0.742 | closed=0.987 progress=0.169 net=0.131 | drift=0.046 dir_err=n/a
  move_turn: n=200 surv=0.995 (move_turn=1.00 move_turn_r=0.99) | ratio=0.789 achieved=0.174 tripod=0.571 | closed=0.571 progress=0.074 net=0.075 | drift=0.190 dir_err=0.099

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.155 action_fft_freq=1.553 joint_speed_mean=0.369 rod_speed_mean=0.029 contact_switch_hz=1.306 foot_slip=0.082 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.028 settle_six_down_s=12.000 foot_disp_end=0.100 work_per_s=378.448
        fwd: action_jitter=0.110 action_fft_freq=1.002 joint_speed_mean=0.757 rod_speed_mean=0.029 contact_switch_hz=1.688 foot_slip=0.108 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=19.621 slip_per_m=1.298 work_per_m=1421.420 cam_work_per_m=540.264
       side: action_jitter=0.086 action_fft_freq=1.040 joint_speed_mean=0.387 rod_speed_mean=0.029 contact_switch_hz=1.243 foot_slip=0.089 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=16.016 slip_per_m=1.237 work_per_m=1411.230 cam_work_per_m=256.836
       back: action_jitter=0.115 action_fft_freq=1.045 joint_speed_mean=0.711 rod_speed_mean=0.029 contact_switch_hz=1.604 foot_slip=0.109 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=21.766 slip_per_m=1.524 work_per_m=1552.282 cam_work_per_m=518.674
       diag: action_jitter=0.102 action_fft_freq=1.108 joint_speed_mean=0.575 rod_speed_mean=0.029 contact_switch_hz=1.653 foot_slip=0.111 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=19.256 slip_per_m=1.321 work_per_m=1288.847 cam_work_per_m=277.588
       turn: action_jitter=0.116 action_fft_freq=1.044 joint_speed_mean=0.687 rod_speed_mean=0.029 contact_switch_hz=1.306 foot_slip=0.090 wz_active_frac=0.710 clock_restarts=0.000 swings_per_rad=30.194 slip_per_rad=2.249 work_per_rad=2407.118 cam_work_per_rad=758.848
  move_turn: action_jitter=0.102 action_fft_freq=1.011 joint_speed_mean=0.681 rod_speed_mean=0.028 contact_switch_hz=1.681 foot_slip=0.118 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=66.452 slip_per_rad=4.793 work_per_rad=5199.277 cam_work_per_rad=2105.049

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.327 steps_per_s=2.375 gait_hz=0.403 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=10.072 support_feet=3.114 taps_per_m=9.406 swing_apex=0.118 step_rate_hz=2.583 gait_hz=0.385 clock_hz=0.357 cadence_ratio=1.077 hz_speed_slope=-0.236 offset_R=0.197
       side: steps_per_m=10.619 support_feet=3.200 taps_per_m=5.477 swing_apex=0.101 step_rate_hz=2.417 gait_hz=0.378 clock_hz=0.314 cadence_ratio=1.204 hz_speed_slope=1.965 offset_R=0.077
       back: steps_per_m=12.078 support_feet=3.162 taps_per_m=9.474 swing_apex=0.108 step_rate_hz=2.667 gait_hz=0.409 clock_hz=0.314 cadence_ratio=1.301 hz_speed_slope=-0.100 offset_R=0.028
       diag: steps_per_m=11.290 support_feet=3.082 taps_per_m=7.700 swing_apex=0.094 step_rate_hz=2.917 gait_hz=0.452 clock_hz=0.357 cadence_ratio=1.265 hz_speed_slope=-0.035 offset_R=0.143
       turn: steps_per_rad=19.381 support_feet=3.256 taps_per_rad=10.690 swing_apex=0.136 step_rate_hz=2.500 gait_hz=0.409 clock_hz=0.352 cadence_ratio=1.167 steps_per_rad_pre=13.463 steps_per_rad_post=5.781 offset_R=0.091
  move_turn: steps_per_rad=33.518 support_feet=3.101 taps_per_rad=30.889 swing_apex=0.106 step_rate_hz=2.583 gait_hz=0.366 clock_hz=0.500 cadence_ratio=0.732 hz_speed_slope=-0.365 offset_R=0.042

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
