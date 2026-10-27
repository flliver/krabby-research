```
=== crab-hex gait eval ===
scenario   : suite__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_03-27-53_seed19_X07_p3a_A15pB/model_22993.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6727751392657428  p25=0.6120423863851375  p75=0.734788373445842
tippy_tap_fraction median=0.4246575342465753
slip_ratio         median=0.20825942647718915
tracking_ratio     median=0.9932404392032305  (achieved/commanded vx, walking holds; n=700)
schedule_completion_rate=0.995
terminations={'schedule_complete': 995, 'fall': 5}
facing_error_rms   median=0.05377942729216706  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.30631488790841055 (n=320)
     stand: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.6759244361231256 (n=100)
       fwd: cmd 0.25 -> achieved 0.254 m/s | tripod median=0.6495129168236448 (n=100)
     lat_l: cmd 0.22 -> achieved 0.213 m/s | tripod median=0.6763254295131564 (n=100)
     lat_r: cmd 0.22 -> achieved 0.241 m/s | tripod median=0.7497845426239927 (n=100)
      back: cmd 0.22 -> achieved 0.230 m/s | tripod median=0.6813315535524336 (n=100)
      diag: cmd 0.25 -> achieved 0.248 m/s | tripod median=0.6299322886851892 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.002 m/s | tripod median=0.7751983633361159 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.774125101683611 (n=100)
  move_turn: cmd 0.22 -> achieved 0.176 m/s | tripod median=0.49268476126921007 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.149 m/s | tripod median=0.6522315268993241 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.010 initial=0.007 final=0.011 max=0.022 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.003 yaw_rate_ratio=n/a
     stand: err rms=0.020 initial=0.011 final=0.021 max=0.049 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.050 initial=0.012 final=0.058 max=0.106 closed=n/a | settle=0.000 (n=100/100) | progress=-0.001 net=-0.000 yaw_rate_ratio=0.336
     lat_l: err rms=0.022 initial=0.010 final=0.018 max=0.051 closed=n/a | settle=0.000 (n=100/100) | progress=-0.005 net=0.002 yaw_rate_ratio=0.314
     lat_r: err rms=0.015 initial=0.011 final=0.009 max=0.035 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
      back: err rms=0.066 initial=0.013 final=0.115 max=0.145 closed=n/a | settle=0.000 (n=90/100) | progress=-0.004 net=-0.000 yaw_rate_ratio=0.368
      diag: err rms=0.034 initial=0.011 final=0.019 max=0.084 closed=n/a | settle=0.000 (n=98/100) | progress=-0.004 net=0.001 yaw_rate_ratio=0.446
  turn_l90: err rms=0.855 initial=1.568 final=0.127 max=1.572 closed=0.919 | settle=10.940 (n=81/100) | progress=0.132 net=0.120 yaw_rate_ratio=0.561
  turn_r90: err rms=0.866 initial=1.579 final=0.134 max=1.583 closed=0.915 | settle=10.940 (n=74/100) | progress=0.130 net=0.121 yaw_rate_ratio=0.549
  move_turn: err rms=1.191 initial=1.567 final=0.748 max=1.584 closed=0.521 | settle=n/a (n=0/100) | progress=0.068 net=0.068 yaw_rate_ratio=0.425
  move_turn_r: err rms=1.163 initial=1.579 final=0.788 max=1.582 closed=0.503 | settle=n/a (n=0/100) | progress=0.066 net=0.066 yaw_rate_ratio=0.439

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.002 tripod=0.676 | closed=n/a progress=n/a net=0.001 | drift=0.038 dir_err=n/a
        fwd: n=100 surv=1.000 (fwd=1.00) | ratio=1.016 achieved=0.254 tripod=0.650 | closed=n/a progress=-0.001 net=-0.000 | drift=0.255 dir_err=0.030
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=1.036 achieved=0.228 tripod=0.719 | closed=n/a progress=-0.005 net=0.001 | drift=0.230 dir_err=0.012
       back: n=100 surv=1.000 (back=1.00) | ratio=1.045 achieved=0.230 tripod=0.681 | closed=n/a progress=-0.004 net=-0.000 | drift=0.230 dir_err=0.039
       diag: n=100 surv=0.990 (diag=0.99) | ratio=0.992 achieved=0.248 tripod=0.630 | closed=n/a progress=-0.004 net=0.001 | drift=0.250 dir_err=0.025
       turn: n=200 surv=1.000 (turn_l90=1.00 turn_r90=1.00) | ratio=n/a achieved=0.001 tripod=0.775 | closed=0.916 progress=0.132 net=0.120 | drift=0.047 dir_err=n/a
  move_turn: n=200 surv=0.980 (move_turn=0.96 move_turn_r=1.00) | ratio=0.729 achieved=0.160 tripod=0.606 | closed=0.514 progress=0.067 net=0.067 | drift=0.176 dir_err=0.085

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.165 action_fft_freq=1.594 joint_speed_mean=0.337 rod_speed_mean=0.029 contact_switch_hz=1.625 foot_slip=0.083 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.004 settle_six_down_s=12.000 foot_disp_end=0.221 work_per_s=303.855
        fwd: action_jitter=0.136 action_fft_freq=1.168 joint_speed_mean=0.721 rod_speed_mean=0.029 contact_switch_hz=1.764 foot_slip=0.109 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=21.081 slip_per_m=1.311 work_per_m=1319.549 cam_work_per_m=516.720
       side: action_jitter=0.103 action_fft_freq=1.136 joint_speed_mean=0.364 rod_speed_mean=0.028 contact_switch_hz=1.236 foot_slip=0.083 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=17.216 slip_per_m=1.172 work_per_m=1288.828 cam_work_per_m=187.958
       back: action_jitter=0.136 action_fft_freq=1.181 joint_speed_mean=0.699 rod_speed_mean=0.029 contact_switch_hz=1.736 foot_slip=0.104 wz_active_frac=0.027 clock_restarts=0.000 swings_per_m=22.628 slip_per_m=1.380 work_per_m=1339.492 cam_work_per_m=549.604
       diag: action_jitter=0.119 action_fft_freq=1.196 joint_speed_mean=0.605 rod_speed_mean=0.029 contact_switch_hz=1.819 foot_slip=0.121 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=22.366 slip_per_m=1.558 work_per_m=1244.076 cam_work_per_m=414.328
       turn: action_jitter=0.132 action_fft_freq=1.104 joint_speed_mean=0.689 rod_speed_mean=0.029 contact_switch_hz=1.417 foot_slip=0.087 wz_active_frac=0.912 clock_restarts=1.000 swings_per_rad=35.322 slip_per_rad=2.305 work_per_rad=2255.237 cam_work_per_rad=747.898
  move_turn: action_jitter=0.122 action_fft_freq=1.127 joint_speed_mean=0.638 rod_speed_mean=0.028 contact_switch_hz=1.611 foot_slip=0.120 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=73.301 slip_per_rad=5.390 work_per_rad=5300.176 cam_work_per_rad=2124.999

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.208 steps_per_s=2.583 gait_hz=0.378 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=10.621 support_feet=3.025 taps_per_m=10.239 swing_apex=0.108 step_rate_hz=2.667 gait_hz=0.385 clock_hz=0.357 cadence_ratio=1.077 hz_speed_slope=0.939 offset_R=0.152
       side: steps_per_m=10.566 support_feet=3.181 taps_per_m=6.057 swing_apex=0.103 step_rate_hz=2.333 gait_hz=0.372 clock_hz=0.314 cadence_ratio=1.185 hz_speed_slope=-1.189 offset_R=0.041
       back: steps_per_m=12.069 support_feet=3.023 taps_per_m=10.621 swing_apex=0.107 step_rate_hz=2.750 gait_hz=0.397 clock_hz=0.316 cadence_ratio=1.243 hz_speed_slope=0.656 offset_R=0.087
       diag: steps_per_m=11.936 support_feet=3.171 taps_per_m=9.905 swing_apex=0.087 step_rate_hz=2.917 gait_hz=0.439 clock_hz=0.357 cadence_ratio=1.230 hz_speed_slope=-0.093 offset_R=0.078
       turn: steps_per_rad=21.210 support_feet=3.189 taps_per_rad=13.658 swing_apex=0.119 step_rate_hz=2.583 gait_hz=0.409 clock_hz=0.447 cadence_ratio=0.921 steps_per_rad_pre=18.915 steps_per_rad_post=2.104 offset_R=0.048
  move_turn: steps_per_rad=38.932 support_feet=3.047 taps_per_rad=33.443 swing_apex=0.119 step_rate_hz=2.583 gait_hz=0.369 clock_hz=0.500 cadence_ratio=0.739 hz_speed_slope=1.153 offset_R=0.078

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
