```
=== crab-hex gait eval ===
scenario   : suite__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_28000.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.663723772444992  p25=0.613122940635543  p75=0.7143548985867672
tippy_tap_fraction median=0.3582089552238806
slip_ratio         median=0.18958839873090666
tracking_ratio     median=1.0173173051966589  (achieved/commanded vx, walking holds; n=700)
schedule_completion_rate=1.0
terminations={'schedule_complete': 1000}
facing_error_rms   median=0.056473549435044526  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.5015331494051442 (n=886)
     stand: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7323881972113244 (n=100)
       fwd: cmd 0.25 -> achieved 0.254 m/s | tripod median=0.6318941723086371 (n=100)
     lat_l: cmd 0.22 -> achieved 0.230 m/s | tripod median=0.6500031353254323 (n=100)
     lat_r: cmd 0.22 -> achieved 0.229 m/s | tripod median=0.7396330193345413 (n=100)
      back: cmd 0.22 -> achieved 0.223 m/s | tripod median=0.6810220248007439 (n=100)
      diag: cmd 0.25 -> achieved 0.258 m/s | tripod median=0.6730048085295963 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.011 m/s | tripod median=0.7623573038516103 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.008 m/s | tripod median=0.7601207024173142 (n=100)
  move_turn: cmd 0.22 -> achieved 0.178 m/s | tripod median=0.5229344708735755 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.169 m/s | tripod median=0.6316512348362051 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.012 initial=0.009 final=0.016 max=0.027 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=-0.000 yaw_rate_ratio=n/a
     stand: err rms=0.027 initial=0.015 final=0.026 max=0.057 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.042 initial=0.018 final=0.053 max=0.118 closed=n/a | settle=0.000 (n=97/100) | progress=-0.001 net=0.004 yaw_rate_ratio=0.840
     lat_l: err rms=0.022 initial=0.015 final=0.019 max=0.051 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.002 yaw_rate_ratio=n/a
     lat_r: err rms=0.024 initial=0.017 final=0.027 max=0.056 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
      back: err rms=0.076 initial=0.017 final=0.093 max=0.154 closed=n/a | settle=8.540 (n=97/100) | progress=-0.003 net=-0.005 yaw_rate_ratio=0.723
      diag: err rms=0.035 initial=0.016 final=0.036 max=0.084 closed=n/a | settle=0.000 (n=100/100) | progress=0.016 net=0.002 yaw_rate_ratio=0.557
  turn_l90: err rms=0.789 initial=1.562 final=0.016 max=1.570 closed=0.989 | settle=8.470 (n=100/100) | progress=0.169 net=0.130 yaw_rate_ratio=0.651
  turn_r90: err rms=0.813 initial=1.581 final=0.025 max=1.590 closed=0.984 | settle=8.580 (n=100/100) | progress=0.168 net=0.130 yaw_rate_ratio=0.646
  move_turn: err rms=1.170 initial=1.557 final=0.859 max=1.575 closed=0.451 | settle=n/a (n=0/100) | progress=0.057 net=0.058 yaw_rate_ratio=0.441
  move_turn_r: err rms=1.028 initial=1.583 final=0.397 max=1.588 closed=0.750 | settle=n/a (n=0/100) | progress=0.098 net=0.099 yaw_rate_ratio=0.465

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.003 tripod=0.732 | closed=n/a progress=n/a net=-0.001 | drift=0.030 dir_err=n/a
        fwd: n=100 surv=1.000 (fwd=1.00) | ratio=1.016 achieved=0.254 tripod=0.632 | closed=n/a progress=-0.001 net=0.004 | drift=0.255 dir_err=0.020
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=1.042 achieved=0.229 tripod=0.685 | closed=n/a progress=n/a net=0.002 | drift=0.230 dir_err=0.019
       back: n=100 surv=1.000 (back=1.00) | ratio=1.015 achieved=0.223 tripod=0.681 | closed=n/a progress=-0.003 net=-0.005 | drift=0.224 dir_err=0.021
       diag: n=100 surv=1.000 (diag=1.00) | ratio=1.033 achieved=0.258 tripod=0.673 | closed=n/a progress=0.016 net=0.002 | drift=0.261 dir_err=0.039
       turn: n=200 surv=1.000 (turn_l90=1.00 turn_r90=1.00) | ratio=n/a achieved=-0.002 tripod=0.762 | closed=0.987 progress=0.168 net=0.130 | drift=0.047 dir_err=n/a
  move_turn: n=200 surv=1.000 (move_turn=1.00 move_turn_r=1.00) | ratio=0.788 achieved=0.173 tripod=0.570 | closed=0.585 progress=0.076 net=0.077 | drift=0.187 dir_err=0.105

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.116 action_fft_freq=1.298 joint_speed_mean=0.361 rod_speed_mean=0.028 contact_switch_hz=1.139 foot_slip=0.085 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.033 settle_six_down_s=12.000 foot_disp_end=0.111 work_per_s=406.026
        fwd: action_jitter=0.086 action_fft_freq=0.932 joint_speed_mean=0.756 rod_speed_mean=0.029 contact_switch_hz=1.569 foot_slip=0.111 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=18.828 slip_per_m=1.390 work_per_m=1470.027 cam_work_per_m=610.944
       side: action_jitter=0.068 action_fft_freq=0.957 joint_speed_mean=0.385 rod_speed_mean=0.028 contact_switch_hz=1.194 foot_slip=0.094 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=16.100 slip_per_m=1.318 work_per_m=1440.691 cam_work_per_m=260.766
       back: action_jitter=0.093 action_fft_freq=0.973 joint_speed_mean=0.710 rod_speed_mean=0.029 contact_switch_hz=1.562 foot_slip=0.109 wz_active_frac=0.026 clock_restarts=0.000 swings_per_m=21.328 slip_per_m=1.562 work_per_m=1572.403 cam_work_per_m=549.230
       diag: action_jitter=0.084 action_fft_freq=1.034 joint_speed_mean=0.572 rod_speed_mean=0.029 contact_switch_hz=1.549 foot_slip=0.108 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=18.249 slip_per_m=1.302 work_per_m=1309.493 cam_work_per_m=270.440
       turn: action_jitter=0.092 action_fft_freq=0.961 joint_speed_mean=0.688 rod_speed_mean=0.029 contact_switch_hz=1.208 foot_slip=0.089 wz_active_frac=0.715 clock_restarts=0.000 swings_per_rad=28.185 slip_per_rad=2.209 work_per_rad=2435.434 cam_work_per_rad=813.202
  move_turn: action_jitter=0.080 action_fft_freq=0.940 joint_speed_mean=0.677 rod_speed_mean=0.028 contact_switch_hz=1.611 foot_slip=0.119 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=62.547 slip_per_rad=4.868 work_per_rad=5032.489 cam_work_per_rad=2137.580

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.383 steps_per_s=2.167 gait_hz=0.360 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=10.576 support_feet=3.147 taps_per_m=8.149 swing_apex=0.117 step_rate_hz=2.667 gait_hz=0.385 clock_hz=0.357 cadence_ratio=1.077 hz_speed_slope=-0.476 offset_R=0.153
       side: steps_per_m=10.552 support_feet=3.223 taps_per_m=5.143 swing_apex=0.098 step_rate_hz=2.375 gait_hz=0.375 clock_hz=0.314 cadence_ratio=1.194 hz_speed_slope=3.332 offset_R=0.013
       back: steps_per_m=12.166 support_feet=3.170 taps_per_m=8.934 swing_apex=0.106 step_rate_hz=2.667 gait_hz=0.403 clock_hz=0.316 cadence_ratio=1.275 hz_speed_slope=-0.351 offset_R=0.092
       diag: steps_per_m=11.430 support_feet=3.107 taps_per_m=6.645 swing_apex=0.091 step_rate_hz=2.917 gait_hz=0.439 clock_hz=0.357 cadence_ratio=1.230 hz_speed_slope=-0.194 offset_R=0.100
       turn: steps_per_rad=19.423 support_feet=3.265 taps_per_rad=8.565 swing_apex=0.137 step_rate_hz=2.500 gait_hz=0.409 clock_hz=0.355 cadence_ratio=1.163 steps_per_rad_pre=14.441 steps_per_rad_post=5.346 offset_R=0.031
  move_turn: steps_per_rad=35.521 support_feet=3.124 taps_per_rad=26.074 swing_apex=0.102 step_rate_hz=2.750 gait_hz=0.366 clock_hz=0.500 cadence_ratio=0.732 hz_speed_slope=-0.099 offset_R=0.075

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
