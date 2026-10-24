```
=== crab-hex gait eval ===
scenario   : suite__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-21_23-06-12_seed19_X06_p2a_A15pB/model_19994.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7508268269895064  p25=0.7011463752627245  p75=0.7968996404291241
tippy_tap_fraction median=0.43902439024390244
slip_ratio         median=0.2289710978300541
tracking_ratio     median=1.0398905746553753  (achieved/commanded vx, walking holds; n=700)
schedule_completion_rate=1.0
terminations={'schedule_complete': 1000}
facing_error_rms   median=0.04565237187480205  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.7750888314196822 (n=986)
     stand: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.7950360069636144 (n=100)
       fwd: cmd 0.25 -> achieved 0.265 m/s | tripod median=0.6124884780221516 (n=100)
     lat_l: cmd 0.22 -> achieved 0.234 m/s | tripod median=0.719101242041686 (n=100)
     lat_r: cmd 0.22 -> achieved 0.234 m/s | tripod median=0.7966362033003307 (n=100)
      back: cmd 0.22 -> achieved 0.233 m/s | tripod median=0.7424538271514083 (n=100)
      diag: cmd 0.25 -> achieved 0.250 m/s | tripod median=0.7493001929643976 (n=100)
  turn_l90: cmd 0.00 -> achieved 0.000 m/s | tripod median=0.8117103408303106 (n=100)
  turn_r90: cmd 0.00 -> achieved -0.001 m/s | tripod median=0.8244936019877667 (n=100)
  move_turn: cmd 0.22 -> achieved 0.169 m/s | tripod median=0.6273636611148641 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.177 m/s | tripod median=0.706441475778639 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.018 initial=0.006 final=0.016 max=0.034 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=-0.000 yaw_rate_ratio=n/a
     stand: err rms=0.027 initial=0.014 final=0.029 max=0.058 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a
       fwd: err rms=0.046 initial=0.016 final=0.055 max=0.089 closed=n/a | settle=0.000 (n=99/100) | progress=-0.000 net=-0.001 yaw_rate_ratio=0.283
     lat_l: err rms=0.029 initial=0.016 final=0.030 max=0.055 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a
     lat_r: err rms=0.020 initial=0.015 final=0.018 max=0.044 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
      back: err rms=0.033 initial=0.017 final=0.033 max=0.070 closed=n/a | settle=0.000 (n=100/100) | progress=0.001 net=0.003 yaw_rate_ratio=0.189
      diag: err rms=0.042 initial=0.018 final=0.038 max=0.086 closed=n/a | settle=0.000 (n=100/100) | progress=0.001 net=0.001 yaw_rate_ratio=0.363
  turn_l90: err rms=0.803 initial=1.560 final=0.092 max=1.561 closed=0.940 | settle=10.000 (n=99/100) | progress=0.150 net=0.122 yaw_rate_ratio=0.592
  turn_r90: err rms=0.836 initial=1.584 final=0.107 max=1.587 closed=0.933 | settle=10.270 (n=98/100) | progress=0.144 net=0.123 yaw_rate_ratio=0.587
  move_turn: err rms=1.106 initial=1.560 final=0.597 max=1.561 closed=0.616 | settle=n/a (n=0/100) | progress=0.079 net=0.080 yaw_rate_ratio=0.356
  move_turn_r: err rms=1.128 initial=1.583 final=0.655 max=1.590 closed=0.588 | settle=n/a (n=0/100) | progress=0.077 net=0.077 yaw_rate_ratio=0.386

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.002 tripod=0.795 | closed=n/a progress=n/a net=0.000 | drift=0.038 dir_err=n/a
        fwd: n=100 surv=1.000 (fwd=1.00) | ratio=1.058 achieved=0.265 tripod=0.612 | closed=n/a progress=-0.000 net=-0.001 | drift=0.262 dir_err=0.062
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=1.064 achieved=0.234 tripod=0.761 | closed=n/a progress=n/a net=0.000 | drift=0.233 dir_err=0.016
       back: n=100 surv=1.000 (back=1.00) | ratio=1.059 achieved=0.233 tripod=0.742 | closed=n/a progress=0.001 net=0.003 | drift=0.233 dir_err=0.046
       diag: n=100 surv=1.000 (diag=1.00) | ratio=0.999 achieved=0.250 tripod=0.749 | closed=n/a progress=0.001 net=0.001 | drift=0.252 dir_err=0.038
       turn: n=200 surv=1.000 (turn_l90=1.00 turn_r90=1.00) | ratio=n/a achieved=-0.001 tripod=0.818 | closed=0.936 progress=0.147 net=0.123 | drift=0.048 dir_err=n/a
  move_turn: n=200 surv=1.000 (move_turn=1.00 move_turn_r=1.00) | ratio=0.791 achieved=0.174 tripod=0.670 | closed=0.600 progress=0.078 net=0.079 | drift=0.179 dir_err=0.085

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.206 action_fft_freq=1.926 joint_speed_mean=0.336 rod_speed_mean=0.029 contact_switch_hz=1.611 foot_slip=0.069 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.003 settle_six_down_s=12.000 foot_disp_end=0.085 work_per_s=241.053
        fwd: action_jitter=0.155 action_fft_freq=1.237 joint_speed_mean=0.760 rod_speed_mean=0.029 contact_switch_hz=2.049 foot_slip=0.103 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=23.503 slip_per_m=1.158 work_per_m=1102.227 cam_work_per_m=397.724
       side: action_jitter=0.135 action_fft_freq=1.426 joint_speed_mean=0.345 rod_speed_mean=0.029 contact_switch_hz=1.583 foot_slip=0.074 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=20.342 slip_per_m=0.987 work_per_m=1017.055 cam_work_per_m=203.094
       back: action_jitter=0.166 action_fft_freq=1.280 joint_speed_mean=0.720 rod_speed_mean=0.029 contact_switch_hz=1.833 foot_slip=0.101 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=23.776 slip_per_m=1.289 work_per_m=1131.616 cam_work_per_m=403.864
       diag: action_jitter=0.156 action_fft_freq=1.329 joint_speed_mean=0.613 rod_speed_mean=0.029 contact_switch_hz=1.806 foot_slip=0.083 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=21.945 slip_per_m=1.025 work_per_m=988.113 cam_work_per_m=253.321
       turn: action_jitter=0.151 action_fft_freq=1.243 joint_speed_mean=0.696 rod_speed_mean=0.029 contact_switch_hz=1.458 foot_slip=0.091 wz_active_frac=0.815 clock_restarts=1.000 swings_per_rad=35.847 slip_per_rad=2.326 work_per_rad=2170.049 cam_work_per_rad=652.356
  move_turn: action_jitter=0.155 action_fft_freq=1.285 joint_speed_mean=0.643 rod_speed_mean=0.028 contact_switch_hz=1.972 foot_slip=0.108 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=74.839 slip_per_rad=4.204 work_per_rad=4012.399 cam_work_per_rad=1162.799

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.060 steps_per_s=2.833 gait_hz=0.476 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=9.922 support_feet=2.942 taps_per_m=13.333 swing_apex=0.126 step_rate_hz=2.583 gait_hz=0.378 clock_hz=0.357 cadence_ratio=1.060 hz_speed_slope=0.338 offset_R=0.126
       side: steps_per_m=12.744 support_feet=3.088 taps_per_m=7.403 swing_apex=0.086 step_rate_hz=2.917 gait_hz=0.494 clock_hz=0.314 cadence_ratio=1.573 hz_speed_slope=1.068 offset_R=0.052
       back: steps_per_m=11.102 support_feet=2.970 taps_per_m=12.255 swing_apex=0.124 step_rate_hz=2.583 gait_hz=0.403 clock_hz=0.314 cadence_ratio=1.282 hz_speed_slope=-0.198 offset_R=0.195
       diag: steps_per_m=12.258 support_feet=3.042 taps_per_m=9.547 swing_apex=0.098 step_rate_hz=3.083 gait_hz=0.500 clock_hz=0.357 cadence_ratio=1.401 hz_speed_slope=0.159 offset_R=0.056
       turn: steps_per_rad=20.837 support_feet=3.105 taps_per_rad=14.343 swing_apex=0.131 step_rate_hz=2.583 gait_hz=0.433 clock_hz=0.401 cadence_ratio=1.070 steps_per_rad_pre=16.634 steps_per_rad_post=4.099 offset_R=0.058
  move_turn: steps_per_rad=33.558 support_feet=3.003 taps_per_rad=41.500 swing_apex=0.126 step_rate_hz=2.667 gait_hz=0.372 clock_hz=0.500 cadence_ratio=0.745 hz_speed_slope=-0.571 offset_R=0.039

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
