```
=== crab-hex gait eval ===
scenario   : suite__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_03-27-53_seed19_X07_p3a_A15pB/model_22993.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7126300716774234  p25=0.6649590964163867  p75=0.756721307814596
tippy_tap_fraction median=0.4264705882352941
slip_ratio         median=0.21986215067660178
tracking_ratio     median=0.9809322471147673  (achieved/commanded vx, walking holds; n=700)
schedule_completion_rate=1.0
terminations={'schedule_complete': 1000}
facing_error_rms   median=0.056436087447986845  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.33438715993182516 (n=301)
     stand: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7822428998727489 (n=100)
       fwd: cmd 0.25 -> achieved 0.252 m/s | tripod median=0.7060602753844029 (n=100)
     lat_l: cmd 0.22 -> achieved 0.211 m/s | tripod median=0.7116508269919052 (n=100)
     lat_r: cmd 0.22 -> achieved 0.244 m/s | tripod median=0.7553915169747238 (n=100)
      back: cmd 0.22 -> achieved 0.228 m/s | tripod median=0.7347208265807497 (n=100)
      diag: cmd 0.25 -> achieved 0.244 m/s | tripod median=0.6831795602718078 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.002 m/s | tripod median=0.8011706054481136 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.7978535026782647 (n=100)
  move_turn: cmd 0.22 -> achieved 0.170 m/s | tripod median=0.6004828079966016 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.142 m/s | tripod median=0.6704523928161252 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.011 initial=0.007 final=0.012 max=0.022 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.002 yaw_rate_ratio=n/a
     stand: err rms=0.022 initial=0.012 final=0.018 max=0.053 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a
       fwd: err rms=0.051 initial=0.012 final=0.063 max=0.114 closed=n/a | settle=0.000 (n=95/100) | progress=-0.005 net=-0.001 yaw_rate_ratio=0.332
     lat_l: err rms=0.021 initial=0.011 final=0.017 max=0.049 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
     lat_r: err rms=0.014 initial=0.013 final=0.010 max=0.035 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
      back: err rms=0.075 initial=0.012 final=0.127 max=0.147 closed=n/a | settle=0.000 (n=89/100) | progress=-0.003 net=0.008 yaw_rate_ratio=0.308
      diag: err rms=0.033 initial=0.009 final=0.020 max=0.081 closed=n/a | settle=0.000 (n=100/100) | progress=-0.013 net=0.001 yaw_rate_ratio=0.306
  turn_l90: err rms=0.838 initial=1.563 final=0.126 max=1.569 closed=0.920 | settle=10.650 (n=86/100) | progress=0.139 net=0.120 yaw_rate_ratio=0.578
  turn_r90: err rms=0.847 initial=1.575 final=0.124 max=1.580 closed=0.921 | settle=10.780 (n=87/100) | progress=0.133 net=0.120 yaw_rate_ratio=0.570
  move_turn: err rms=1.130 initial=1.567 final=0.593 max=1.580 closed=0.617 | settle=n/a (n=0/100) | progress=0.081 net=0.081 yaw_rate_ratio=0.399
  move_turn_r: err rms=1.167 initial=1.578 final=0.772 max=1.582 closed=0.511 | settle=n/a (n=0/100) | progress=0.067 net=0.067 yaw_rate_ratio=0.441

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.003 tripod=0.782 | closed=n/a progress=n/a net=0.000 | drift=0.038 dir_err=n/a
        fwd: n=100 surv=1.000 (fwd=1.00) | ratio=1.007 achieved=0.252 tripod=0.706 | closed=n/a progress=-0.005 net=-0.001 | drift=0.253 dir_err=0.033
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=1.031 achieved=0.227 tripod=0.736 | closed=n/a progress=n/a net=0.001 | drift=0.226 dir_err=0.012
       back: n=100 surv=1.000 (back=1.00) | ratio=1.035 achieved=0.228 tripod=0.735 | closed=n/a progress=-0.003 net=0.008 | drift=0.228 dir_err=0.045
       diag: n=100 surv=1.000 (diag=1.00) | ratio=0.977 achieved=0.244 tripod=0.683 | closed=n/a progress=-0.013 net=0.001 | drift=0.245 dir_err=0.016
       turn: n=200 surv=1.000 (turn_l90=1.00 turn_r90=1.00) | ratio=n/a achieved=0.001 tripod=0.799 | closed=0.920 progress=0.136 net=0.120 | drift=0.049 dir_err=n/a
  move_turn: n=200 surv=1.000 (move_turn=1.00 move_turn_r=1.00) | ratio=0.714 achieved=0.157 tripod=0.647 | closed=0.557 progress=0.073 net=0.073 | drift=0.172 dir_err=0.113

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.173 action_fft_freq=1.623 joint_speed_mean=0.335 rod_speed_mean=0.029 contact_switch_hz=1.417 foot_slip=0.077 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.003 settle_six_down_s=12.000 foot_disp_end=0.220 work_per_s=275.059
        fwd: action_jitter=0.145 action_fft_freq=1.244 joint_speed_mean=0.718 rod_speed_mean=0.029 contact_switch_hz=1.799 foot_slip=0.102 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=21.656 slip_per_m=1.216 work_per_m=1170.627 cam_work_per_m=455.529
       side: action_jitter=0.113 action_fft_freq=1.226 joint_speed_mean=0.364 rod_speed_mean=0.028 contact_switch_hz=1.319 foot_slip=0.076 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=17.931 slip_per_m=1.047 work_per_m=1123.370 cam_work_per_m=181.340
       back: action_jitter=0.145 action_fft_freq=1.241 joint_speed_mean=0.690 rod_speed_mean=0.029 contact_switch_hz=1.778 foot_slip=0.093 wz_active_frac=0.049 clock_restarts=0.000 swings_per_m=23.614 slip_per_m=1.229 work_per_m=1191.464 cam_work_per_m=469.595
       diag: action_jitter=0.129 action_fft_freq=1.268 joint_speed_mean=0.598 rod_speed_mean=0.029 contact_switch_hz=1.847 foot_slip=0.097 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=22.886 slip_per_m=1.244 work_per_m=1086.507 cam_work_per_m=315.650
       turn: action_jitter=0.139 action_fft_freq=1.153 joint_speed_mean=0.684 rod_speed_mean=0.029 contact_switch_hz=1.444 foot_slip=0.082 wz_active_frac=0.870 clock_restarts=1.000 swings_per_rad=36.005 slip_per_rad=2.137 work_per_rad=2082.418 cam_work_per_rad=674.430
  move_turn: action_jitter=0.128 action_fft_freq=1.170 joint_speed_mean=0.635 rod_speed_mean=0.028 contact_switch_hz=1.667 foot_slip=0.113 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=67.965 slip_per_rad=4.752 work_per_rad=4484.953 cam_work_per_rad=1656.891

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.114 steps_per_s=2.417 gait_hz=0.372 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=10.582 support_feet=2.984 taps_per_m=10.754 swing_apex=0.111 step_rate_hz=2.667 gait_hz=0.397 clock_hz=0.357 cadence_ratio=1.111 hz_speed_slope=0.454 offset_R=0.145
       side: steps_per_m=10.777 support_feet=3.126 taps_per_m=6.746 swing_apex=0.104 step_rate_hz=2.333 gait_hz=0.385 clock_hz=0.314 cadence_ratio=1.223 hz_speed_slope=-1.009 offset_R=0.068
       back: steps_per_m=11.979 support_feet=2.980 taps_per_m=11.291 swing_apex=0.113 step_rate_hz=2.750 gait_hz=0.403 clock_hz=0.317 cadence_ratio=1.262 hz_speed_slope=0.295 offset_R=0.076
       diag: steps_per_m=12.511 support_feet=3.071 taps_per_m=10.175 swing_apex=0.089 step_rate_hz=3.000 gait_hz=0.446 clock_hz=0.357 cadence_ratio=1.248 hz_speed_slope=-0.331 offset_R=0.143
       turn: steps_per_rad=21.170 support_feet=3.133 taps_per_rad=14.476 swing_apex=0.122 step_rate_hz=2.583 gait_hz=0.415 clock_hz=0.426 cadence_ratio=0.969 steps_per_rad_pre=18.406 steps_per_rad_post=2.749 offset_R=0.041
  move_turn: steps_per_rad=35.512 support_feet=3.012 taps_per_rad=32.427 swing_apex=0.121 step_rate_hz=2.583 gait_hz=0.366 clock_hz=0.500 cadence_ratio=0.732 hz_speed_slope=0.855 offset_R=0.058

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
