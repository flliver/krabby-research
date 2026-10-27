```
=== crab-hex gait eval ===
scenario   : suite__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_30000.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7307935725130734  p25=0.6243060017461952  p75=0.7898833138618145
tippy_tap_fraction median=0.40540540540540543
slip_ratio         median=0.23069134403658478
tracking_ratio     median=0.9789840732321062  (achieved/commanded vx, walking holds; n=700)
schedule_completion_rate=1.0
terminations={'schedule_complete': 1000}
facing_error_rms   median=0.07139050543999442  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved -0.003 m/s | tripod median=0.7898594676205207 (n=997)
     stand: cmd 0.00 -> achieved -0.000 m/s | tripod median=0.7978793952254358 (n=100)
       fwd: cmd 0.25 -> achieved 0.235 m/s | tripod median=0.6105920198996321 (n=100)
     lat_l: cmd 0.22 -> achieved 0.226 m/s | tripod median=0.6172651092979612 (n=100)
     lat_r: cmd 0.22 -> achieved 0.220 m/s | tripod median=0.7847000358294636 (n=100)
      back: cmd 0.22 -> achieved 0.218 m/s | tripod median=0.7547135321425846 (n=100)
      diag: cmd 0.25 -> achieved 0.247 m/s | tripod median=0.6067597521109827 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.008 m/s | tripod median=0.7987047891675099 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.005 m/s | tripod median=0.8010269286083245 (n=100)
  move_turn: cmd 0.22 -> achieved 0.135 m/s | tripod median=0.47789770080798033 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.174 m/s | tripod median=0.7088523621464415 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.022 initial=0.016 final=0.026 max=0.036 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=-0.004 yaw_rate_ratio=n/a
     stand: err rms=0.038 initial=0.026 final=0.044 max=0.063 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.073 initial=0.028 final=0.040 max=0.157 closed=n/a | settle=3.910 (n=100/100) | progress=0.000 net=0.000 yaw_rate_ratio=0.652
     lat_l: err rms=0.026 initial=0.027 final=0.025 max=0.054 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
     lat_r: err rms=0.055 initial=0.025 final=0.068 max=0.091 closed=n/a | settle=0.000 (n=100/100) | progress=-0.002 net=-0.003 yaw_rate_ratio=0.563
      back: err rms=0.081 initial=0.025 final=0.129 max=0.150 closed=n/a | settle=0.000 (n=86/100) | progress=-0.004 net=-0.008 yaw_rate_ratio=0.304
      diag: err rms=0.044 initial=0.026 final=0.018 max=0.098 closed=n/a | settle=0.000 (n=100/100) | progress=0.002 net=0.002 yaw_rate_ratio=0.293
  turn_l90: err rms=0.772 initial=1.543 final=0.018 max=1.546 closed=0.989 | settle=7.990 (n=100/100) | progress=0.174 net=0.128 yaw_rate_ratio=0.628
  turn_r90: err rms=0.795 initial=1.596 final=0.060 max=1.599 closed=0.963 | settle=8.190 (n=100/100) | progress=0.177 net=0.128 yaw_rate_ratio=0.655
  move_turn: err rms=1.110 initial=1.545 final=0.781 max=1.547 closed=0.492 | settle=n/a (n=0/100) | progress=0.062 net=0.063 yaw_rate_ratio=0.406
  move_turn_r: err rms=1.063 initial=1.597 final=0.323 max=1.607 closed=0.798 | settle=n/a (n=0/100) | progress=0.107 net=0.107 yaw_rate_ratio=0.453

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=-0.000 tripod=0.798 | closed=n/a progress=n/a net=-0.001 | drift=0.036 dir_err=n/a
        fwd: n=100 surv=1.000 (fwd=1.00) | ratio=0.942 achieved=0.235 tripod=0.611 | closed=n/a progress=0.000 net=0.000 | drift=0.238 dir_err=0.056
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=1.014 achieved=0.223 tripod=0.697 | closed=n/a progress=-0.002 net=-0.001 | drift=0.224 dir_err=0.011
       back: n=100 surv=1.000 (back=1.00) | ratio=0.990 achieved=0.218 tripod=0.755 | closed=n/a progress=-0.004 net=-0.008 | drift=0.219 dir_err=0.043
       diag: n=100 surv=1.000 (diag=1.00) | ratio=0.990 achieved=0.247 tripod=0.607 | closed=n/a progress=0.002 net=0.002 | drift=0.250 dir_err=0.115
       turn: n=200 surv=1.000 (turn_l90=1.00 turn_r90=1.00) | ratio=n/a achieved=-0.001 tripod=0.800 | closed=0.981 progress=0.175 net=0.128 | drift=0.048 dir_err=n/a
  move_turn: n=200 surv=1.000 (move_turn=1.00 move_turn_r=1.00) | ratio=0.729 achieved=0.160 tripod=0.634 | closed=0.654 progress=0.084 net=0.085 | drift=0.176 dir_err=0.111

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.136 action_fft_freq=1.628 joint_speed_mean=0.371 rod_speed_mean=0.029 contact_switch_hz=1.653 foot_slip=0.068 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.012 settle_six_down_s=12.000 foot_disp_end=0.119 work_per_s=273.396
        fwd: action_jitter=0.107 action_fft_freq=1.117 joint_speed_mean=0.699 rod_speed_mean=0.029 contact_switch_hz=2.042 foot_slip=0.098 wz_active_frac=0.037 clock_restarts=0.000 swings_per_m=25.908 slip_per_m=1.238 work_per_m=1174.490 cam_work_per_m=462.018
       side: action_jitter=0.082 action_fft_freq=1.085 joint_speed_mean=0.363 rod_speed_mean=0.028 contact_switch_hz=1.444 foot_slip=0.078 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=19.262 slip_per_m=1.100 work_per_m=1081.831 cam_work_per_m=212.083
       back: action_jitter=0.119 action_fft_freq=1.183 joint_speed_mean=0.687 rod_speed_mean=0.029 contact_switch_hz=1.694 foot_slip=0.085 wz_active_frac=0.078 clock_restarts=0.000 swings_per_m=23.468 slip_per_m=1.179 work_per_m=1146.541 cam_work_per_m=347.865
       diag: action_jitter=0.094 action_fft_freq=1.159 joint_speed_mean=0.529 rod_speed_mean=0.028 contact_switch_hz=2.028 foot_slip=0.096 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=24.819 slip_per_m=1.160 work_per_m=1033.257 cam_work_per_m=222.741
       turn: action_jitter=0.112 action_fft_freq=1.153 joint_speed_mean=0.668 rod_speed_mean=0.029 contact_switch_hz=1.604 foot_slip=0.075 wz_active_frac=0.688 clock_restarts=0.000 swings_per_rad=37.438 slip_per_rad=1.821 work_per_rad=1963.437 cam_work_per_rad=565.867
  move_turn: action_jitter=0.089 action_fft_freq=1.011 joint_speed_mean=0.665 rod_speed_mean=0.028 contact_switch_hz=1.931 foot_slip=0.102 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=66.279 slip_per_rad=3.610 work_per_rad=3544.903 cam_work_per_rad=1320.620

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.164 steps_per_s=3.083 gait_hz=0.500 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=12.990 support_feet=2.972 taps_per_m=12.709 swing_apex=0.110 step_rate_hz=3.083 gait_hz=0.421 clock_hz=0.358 cadence_ratio=1.165 hz_speed_slope=-0.558 offset_R=0.077
       side: steps_per_m=11.396 support_feet=3.072 taps_per_m=7.856 swing_apex=0.114 step_rate_hz=2.542 gait_hz=0.400 clock_hz=0.314 cadence_ratio=1.272 hz_speed_slope=4.307 offset_R=0.017
       back: steps_per_m=13.106 support_feet=3.018 taps_per_m=10.247 swing_apex=0.116 step_rate_hz=2.833 gait_hz=0.446 clock_hz=0.318 cadence_ratio=1.395 hz_speed_slope=0.170 offset_R=0.143
       diag: steps_per_m=13.700 support_feet=3.008 taps_per_m=10.776 swing_apex=0.077 step_rate_hz=3.417 gait_hz=0.476 clock_hz=0.357 cadence_ratio=1.333 hz_speed_slope=-0.716 offset_R=0.111
       turn: steps_per_rad=21.623 support_feet=3.118 taps_per_rad=16.058 swing_apex=0.142 step_rate_hz=2.750 gait_hz=0.436 clock_hz=0.342 cadence_ratio=1.290 steps_per_rad_pre=14.308 steps_per_rad_post=7.460 offset_R=0.057
  move_turn: steps_per_rad=33.795 support_feet=2.992 taps_per_rad=32.307 swing_apex=0.100 step_rate_hz=2.917 gait_hz=0.391 clock_hz=0.500 cadence_ratio=0.781 hz_speed_slope=-0.393 offset_R=0.046

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
