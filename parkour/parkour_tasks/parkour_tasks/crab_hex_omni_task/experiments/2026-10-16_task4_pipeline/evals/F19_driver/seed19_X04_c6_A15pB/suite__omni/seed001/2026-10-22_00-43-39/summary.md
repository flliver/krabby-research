```
=== crab-hex gait eval ===
scenario   : suite__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-10-21_19-57-38_seed19_X04_c6_A15pB/model_10996.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6769923698030593  p25=0.5025436906007423  p75=0.7522404302297152
tippy_tap_fraction median=0.46440092165898617
slip_ratio         median=0.26181705780478043
tracking_ratio     median=0.9185035175962106  (achieved/commanded vx, walking holds; n=700)
schedule_completion_rate=0.899
terminations={'schedule_complete': 899, 'fall': 101}
facing_error_rms   median=0.054792548541521956  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.000 m/s | tripod median=0.526697300489605 (n=919)
     stand: cmd 0.00 -> achieved 0.000 m/s | tripod median=0.5104030314260849 (n=100)
       fwd: cmd 0.25 -> achieved 0.224 m/s | tripod median=0.6112925665048813 (n=100)
     lat_l: cmd 0.22 -> achieved 0.219 m/s | tripod median=0.8152178576803477 (n=100)
     lat_r: cmd 0.22 -> achieved 0.219 m/s | tripod median=0.7694782483767542 (n=100)
      back: cmd 0.22 -> achieved 0.203 m/s | tripod median=0.6056842757361229 (n=100)
      diag: cmd 0.25 -> achieved 0.233 m/s | tripod median=0.7516981119069539 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.025 m/s | tripod median=0.7656717729481577 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.035 m/s | tripod median=0.7967533698537113 (n=100)
  move_turn: cmd 0.22 -> achieved 0.115 m/s | tripod median=0.37246182688405594 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.162 m/s | tripod median=0.45996497125299984 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.008 initial=0.005 final=0.008 max=0.014 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.000 yaw_rate_ratio=n/a
     stand: err rms=0.012 initial=0.008 final=0.013 max=0.025 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.000 yaw_rate_ratio=n/a
       fwd: err rms=0.030 initial=0.007 final=0.030 max=0.077 closed=n/a | settle=0.000 (n=99/100) | progress=-0.000 net=0.001 yaw_rate_ratio=0.555
     lat_l: err rms=0.017 initial=0.008 final=0.014 max=0.045 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
     lat_r: err rms=0.024 initial=0.007 final=0.024 max=0.054 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.000 yaw_rate_ratio=n/a
      back: err rms=0.050 initial=0.009 final=0.068 max=0.106 closed=n/a | settle=0.000 (n=100/100) | progress=-0.002 net=-0.004 yaw_rate_ratio=0.751
      diag: err rms=0.070 initial=0.009 final=0.115 max=0.128 closed=n/a | settle=0.000 (n=99/100) | progress=-0.002 net=0.010 yaw_rate_ratio=0.306
  turn_l90: err rms=1.271 initial=1.567 final=1.117 max=1.567 closed=0.286 | settle=n/a (n=0/100) | progress=0.038 net=0.037 yaw_rate_ratio=0.301
  turn_r90: err rms=1.173 initial=1.573 final=0.854 max=1.574 closed=0.457 | settle=n/a (n=0/100) | progress=0.061 net=0.060 yaw_rate_ratio=0.289
  move_turn: err rms=1.377 initial=1.568 final=1.332 max=1.570 closed=0.150 | settle=n/a (n=0/100) | progress=0.032 net=0.034 yaw_rate_ratio=0.403
  move_turn_r: err rms=1.273 initial=1.576 final=1.125 max=1.577 closed=0.288 | settle=n/a (n=0/100) | progress=0.038 net=0.039 yaw_rate_ratio=0.418

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.000 tripod=0.510 | closed=n/a progress=n/a net=-0.000 | drift=0.033 dir_err=n/a
        fwd: n=100 surv=1.000 (fwd=1.00) | ratio=0.897 achieved=0.224 tripod=0.611 | closed=n/a progress=-0.000 net=0.001 | drift=0.224 dir_err=0.010
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=0.995 achieved=0.219 tripod=0.787 | closed=n/a progress=n/a net=0.000 | drift=0.219 dir_err=0.012
       back: n=100 surv=1.000 (back=1.00) | ratio=0.921 achieved=0.203 tripod=0.606 | closed=n/a progress=-0.002 net=-0.004 | drift=0.203 dir_err=0.028
       diag: n=100 surv=1.000 (diag=1.00) | ratio=0.932 achieved=0.233 tripod=0.752 | closed=n/a progress=-0.002 net=0.010 | drift=0.233 dir_err=0.038
       turn: n=200 surv=1.000 (turn_l90=1.00 turn_r90=1.00) | ratio=n/a achieved=0.007 tripod=0.781 | closed=0.375 progress=0.049 net=0.049 | drift=0.058 dir_err=n/a
  move_turn: n=200 surv=0.495 (move_turn=0.09 move_turn_r=0.90) | ratio=0.672 achieved=0.148 tripod=0.425 | closed=0.222 progress=0.036 net=0.037 | drift=0.167 dir_err=0.038

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.286 action_fft_freq=3.089 joint_speed_mean=0.283 rod_speed_mean=0.029 contact_switch_hz=3.076 foot_slip=0.054 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.025 settle_six_down_s=12.000 foot_disp_end=0.102 work_per_s=230.446
        fwd: action_jitter=0.234 action_fft_freq=1.774 joint_speed_mean=0.655 rod_speed_mean=0.028 contact_switch_hz=2.097 foot_slip=0.103 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=28.026 slip_per_m=1.334 work_per_m=1084.938 cam_work_per_m=457.451
       side: action_jitter=0.154 action_fft_freq=1.249 joint_speed_mean=0.328 rod_speed_mean=0.028 contact_switch_hz=1.292 foot_slip=0.101 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=18.075 slip_per_m=1.461 work_per_m=1253.923 cam_work_per_m=315.527
       back: action_jitter=0.256 action_fft_freq=2.031 joint_speed_mean=0.591 rod_speed_mean=0.028 contact_switch_hz=2.035 foot_slip=0.093 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=30.486 slip_per_m=1.326 work_per_m=1193.100 cam_work_per_m=443.026
       diag: action_jitter=0.194 action_fft_freq=1.421 joint_speed_mean=0.519 rod_speed_mean=0.029 contact_switch_hz=1.875 foot_slip=0.096 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=24.566 slip_per_m=1.241 work_per_m=1105.496 cam_work_per_m=282.820
       turn: action_jitter=0.199 action_fft_freq=1.599 joint_speed_mean=0.592 rod_speed_mean=0.029 contact_switch_hz=1.715 foot_slip=0.080 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=102.483 slip_per_rad=5.162 work_per_rad=4936.887 cam_work_per_rad=1034.534
  move_turn: action_jitter=0.168 action_fft_freq=1.449 joint_speed_mean=0.656 rod_speed_mean=0.028 contact_switch_hz=2.306 foot_slip=0.122 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=192.616 slip_per_rad=9.544 work_per_rad=7838.551 cam_work_per_rad=3150.978

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.376 steps_per_s=5.083 gait_hz=1.187 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=11.714 support_feet=2.893 taps_per_m=16.250 swing_apex=0.095 step_rate_hz=2.583 gait_hz=0.360 clock_hz=0.357 cadence_ratio=1.008 hz_speed_slope=0.047 offset_R=0.997
       side: steps_per_m=9.395 support_feet=3.102 taps_per_m=8.534 swing_apex=0.110 step_rate_hz=2.000 gait_hz=0.311 clock_hz=0.314 cadence_ratio=0.990 hz_speed_slope=0.014 offset_R=0.996
       back: steps_per_m=12.438 support_feet=2.870 taps_per_m=18.120 swing_apex=0.096 step_rate_hz=2.500 gait_hz=0.317 clock_hz=0.314 cadence_ratio=1.010 hz_speed_slope=0.026 offset_R=0.998
       diag: steps_per_m=10.819 support_feet=2.990 taps_per_m=13.720 swing_apex=0.088 step_rate_hz=2.500 gait_hz=0.360 clock_hz=0.357 cadence_ratio=1.008 hz_speed_slope=0.037 offset_R=0.999
       turn: steps_per_rad=66.069 support_feet=3.091 taps_per_rad=40.727 swing_apex=0.105 step_rate_hz=3.083 gait_hz=0.500 clock_hz=0.500 cadence_ratio=1.001 steps_per_rad_pre=66.069 steps_per_rad_post=n/a offset_R=0.998
  move_turn: steps_per_rad=96.384 support_feet=2.860 taps_per_rad=96.335 swing_apex=0.079 step_rate_hz=3.500 gait_hz=0.494 clock_hz=0.500 cadence_ratio=0.989 hz_speed_slope=0.082 offset_R=0.995

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
