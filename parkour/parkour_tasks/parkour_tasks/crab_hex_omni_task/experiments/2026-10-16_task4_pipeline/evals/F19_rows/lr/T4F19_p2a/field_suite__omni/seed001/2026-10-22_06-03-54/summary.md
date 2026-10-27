```
=== crab-hex gait eval ===
scenario   : field_suite__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-21_23-06-12_seed19_X06_p2a_A15pB/model_19994.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7420310426168311  p25=0.6855526023951314  p75=0.8005461471923816
tippy_tap_fraction median=0.4358974358974359
slip_ratio         median=0.22240634402523793
tracking_ratio     median=0.9898473686436033  (achieved/commanded vx, walking holds; n=700)
schedule_completion_rate=0.998
terminations={'schedule_complete': 998, 'fall': 2}
facing_error_rms   median=0.04076037017190786  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7994327353115696 (n=955)
     stand: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.8146773424805961 (n=100)
       fwd: cmd 0.25 -> achieved 0.258 m/s | tripod median=0.5700198248670727 (n=100)
     lat_l: cmd 0.22 -> achieved 0.220 m/s | tripod median=0.711536701745765 (n=100)
     lat_r: cmd 0.22 -> achieved 0.238 m/s | tripod median=0.7601987357074176 (n=100)
      back: cmd 0.22 -> achieved 0.232 m/s | tripod median=0.7285927647993038 (n=100)
      diag: cmd 0.25 -> achieved 0.238 m/s | tripod median=0.7362622330592142 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.001 m/s | tripod median=0.8184816553682599 (n=100)
  turn_r90: cmd 0.00 -> achieved -0.002 m/s | tripod median=0.8262323554898517 (n=100)
  move_turn: cmd 0.22 -> achieved 0.167 m/s | tripod median=0.5997707461799973 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.172 m/s | tripod median=0.7154323219293847 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.016 initial=0.006 final=0.014 max=0.033 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.001 yaw_rate_ratio=n/a
     stand: err rms=0.022 initial=0.017 final=0.024 max=0.052 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.039 initial=0.014 final=0.040 max=0.083 closed=n/a | settle=0.000 (n=94/100) | progress=-0.010 net=-0.001 yaw_rate_ratio=0.708
     lat_l: err rms=0.025 initial=0.017 final=0.025 max=0.052 closed=n/a | settle=0.000 (n=100/100) | progress=0.002 net=0.001 yaw_rate_ratio=0.447
     lat_r: err rms=0.022 initial=0.014 final=0.023 max=0.045 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
      back: err rms=0.034 initial=0.014 final=0.034 max=0.077 closed=n/a | settle=0.000 (n=100/100) | progress=0.040 net=0.003 yaw_rate_ratio=1.410
      diag: err rms=0.039 initial=0.013 final=0.034 max=0.082 closed=n/a | settle=0.000 (n=99/100) | progress=-0.013 net=0.001 yaw_rate_ratio=0.686
  turn_l90: err rms=0.804 initial=1.566 final=0.101 max=1.567 closed=0.936 | settle=9.900 (n=97/100) | progress=0.153 net=0.122 yaw_rate_ratio=0.597
  turn_r90: err rms=0.813 initial=1.577 final=0.107 max=1.580 closed=0.933 | settle=11.340 (n=97/100) | progress=0.141 net=0.123 yaw_rate_ratio=0.589
  move_turn: err rms=1.151 initial=1.568 final=0.705 max=1.569 closed=0.552 | settle=n/a (n=0/100) | progress=0.072 net=0.072 yaw_rate_ratio=0.349
  move_turn_r: err rms=1.087 initial=1.576 final=0.564 max=1.583 closed=0.639 | settle=n/a (n=0/100) | progress=0.084 net=0.084 yaw_rate_ratio=0.405

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.001 tripod=0.815 | closed=n/a progress=n/a net=0.001 | drift=0.036 dir_err=n/a
        fwd: n=100 surv=0.990 (fwd=0.99) | ratio=1.031 achieved=0.258 tripod=0.570 | closed=n/a progress=-0.010 net=-0.001 | drift=0.258 dir_err=0.109
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=1.022 achieved=0.225 tripod=0.733 | closed=n/a progress=0.002 net=0.001 | drift=0.225 dir_err=0.020
       back: n=100 surv=1.000 (back=1.00) | ratio=1.057 achieved=0.232 tripod=0.729 | closed=n/a progress=0.040 net=0.003 | drift=0.233 dir_err=0.096
       diag: n=100 surv=0.990 (diag=0.99) | ratio=0.951 achieved=0.238 tripod=0.736 | closed=n/a progress=-0.013 net=0.001 | drift=0.240 dir_err=0.068
       turn: n=200 surv=1.000 (turn_l90=1.00 turn_r90=1.00) | ratio=n/a achieved=-0.002 tripod=0.822 | closed=0.934 progress=0.148 net=0.122 | drift=0.049 dir_err=n/a
  move_turn: n=200 surv=1.000 (move_turn=1.00 move_turn_r=1.00) | ratio=0.775 achieved=0.170 tripod=0.655 | closed=0.593 progress=0.077 net=0.077 | drift=0.178 dir_err=0.129

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.203 action_fft_freq=1.912 joint_speed_mean=0.342 rod_speed_mean=0.029 contact_switch_hz=1.549 foot_slip=0.068 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.007 settle_six_down_s=12.000 foot_disp_end=0.109 work_per_s=249.580
        fwd: action_jitter=0.150 action_fft_freq=1.213 joint_speed_mean=0.760 rod_speed_mean=0.029 contact_switch_hz=2.021 foot_slip=0.111 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=23.580 slip_per_m=1.281 work_per_m=1185.446 cam_work_per_m=433.564
       side: action_jitter=0.133 action_fft_freq=1.409 joint_speed_mean=0.353 rod_speed_mean=0.029 contact_switch_hz=1.646 foot_slip=0.074 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=21.758 slip_per_m=1.004 work_per_m=1092.239 cam_work_per_m=198.188
       back: action_jitter=0.165 action_fft_freq=1.276 joint_speed_mean=0.716 rod_speed_mean=0.029 contact_switch_hz=1.833 foot_slip=0.109 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=23.572 slip_per_m=1.398 work_per_m=1189.900 cam_work_per_m=426.870
       diag: action_jitter=0.155 action_fft_freq=1.338 joint_speed_mean=0.611 rod_speed_mean=0.029 contact_switch_hz=1.806 foot_slip=0.091 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=22.815 slip_per_m=1.168 work_per_m=1071.847 cam_work_per_m=273.813
       turn: action_jitter=0.150 action_fft_freq=1.235 joint_speed_mean=0.689 rod_speed_mean=0.029 contact_switch_hz=1.417 foot_slip=0.094 wz_active_frac=0.817 clock_restarts=2.000 swings_per_rad=35.117 slip_per_rad=2.408 work_per_rad=2282.634 cam_work_per_rad=647.148
  move_turn: action_jitter=0.148 action_fft_freq=1.264 joint_speed_mean=0.642 rod_speed_mean=0.028 contact_switch_hz=1.944 foot_slip=0.109 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=74.165 slip_per_rad=4.210 work_per_rad=4140.511 cam_work_per_rad=1121.644

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.103 steps_per_s=2.833 gait_hz=0.479 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=10.144 support_feet=2.947 taps_per_m=13.609 swing_apex=0.125 step_rate_hz=2.583 gait_hz=0.372 clock_hz=0.357 cadence_ratio=1.042 hz_speed_slope=-0.131 offset_R=0.097
       side: steps_per_m=13.473 support_feet=3.112 taps_per_m=8.269 swing_apex=0.088 step_rate_hz=3.000 gait_hz=0.500 clock_hz=0.314 cadence_ratio=1.592 hz_speed_slope=1.166 offset_R=0.067
       back: steps_per_m=11.148 support_feet=2.985 taps_per_m=12.381 swing_apex=0.124 step_rate_hz=2.583 gait_hz=0.403 clock_hz=0.314 cadence_ratio=1.282 hz_speed_slope=0.066 offset_R=0.070
       diag: steps_per_m=13.047 support_feet=3.052 taps_per_m=9.862 swing_apex=0.101 step_rate_hz=3.083 gait_hz=0.500 clock_hz=0.357 cadence_ratio=1.393 hz_speed_slope=-0.766 offset_R=0.136
       turn: steps_per_rad=20.759 support_feet=3.134 taps_per_rad=14.207 swing_apex=0.129 step_rate_hz=2.500 gait_hz=0.427 clock_hz=0.399 cadence_ratio=1.079 steps_per_rad_pre=16.429 steps_per_rad_post=4.109 offset_R=0.047
  move_turn: steps_per_rad=33.277 support_feet=3.002 taps_per_rad=40.254 swing_apex=0.126 step_rate_hz=2.583 gait_hz=0.372 clock_hz=0.500 cadence_ratio=0.745 hz_speed_slope=0.032 offset_R=0.036

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
