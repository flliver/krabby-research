```
=== crab-hex gait eval ===
scenario   : suite__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-10-21_18-21-11_seed19_X02_c4_A15pB/model_7498.pt
episodes   : 1000  unscored: 3
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6642385167197254  p25=0.5554681060135939  p75=0.7131359621191761
tippy_tap_fraction median=0.46153846153846156
slip_ratio         median=0.28190406777029
tracking_ratio     median=0.9384519606493168  (achieved/commanded vx, walking holds; n=699)
schedule_completion_rate=0.8
terminations={'schedule_complete': 800, 'fall': 200}
facing_error_rms   median=0.05127692851525386  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved -0.000 m/s | tripod median=0.45676253856026394 (n=883)
     stand: cmd 0.00 -> achieved -0.000 m/s | tripod median=0.45151209969734296 (n=100)
       fwd: cmd 0.25 -> achieved 0.235 m/s | tripod median=0.6773893090096252 (n=100)
     lat_l: cmd 0.22 -> achieved 0.211 m/s | tripod median=0.7737843211948963 (n=100)
     lat_r: cmd 0.22 -> achieved 0.214 m/s | tripod median=0.712113352890251 (n=100)
      back: cmd 0.22 -> achieved 0.212 m/s | tripod median=0.7015712596326265 (n=100)
      diag: cmd 0.25 -> achieved 0.226 m/s | tripod median=0.668321023544687 (n=100)
  turn_l90: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.7314516565995284 (n=100)
  turn_r90: cmd 0.00 -> achieved -0.000 m/s | tripod median=0.7846216521912768 (n=100)
  move_turn: cmd 0.22 -> achieved 0.146 m/s | tripod median=0.4226742534462573 (n=98)
  move_turn_r: cmd 0.22 -> achieved 0.176 m/s | tripod median=0.41116494767216916 (n=84)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.007 initial=0.007 final=0.006 max=0.011 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.001 yaw_rate_ratio=n/a
     stand: err rms=0.009 initial=0.006 final=0.008 max=0.019 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a
       fwd: err rms=0.055 initial=0.005 final=0.079 max=0.110 closed=n/a | settle=0.000 (n=99/100) | progress=0.000 net=0.006 yaw_rate_ratio=0.351
     lat_l: err rms=0.025 initial=0.008 final=0.032 max=0.050 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.001 yaw_rate_ratio=n/a
     lat_r: err rms=0.019 initial=0.006 final=0.023 max=0.044 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
      back: err rms=0.061 initial=0.006 final=0.097 max=0.111 closed=n/a | settle=0.000 (n=100/100) | progress=0.001 net=-0.005 yaw_rate_ratio=0.281
      diag: err rms=0.026 initial=0.006 final=0.027 max=0.064 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
  turn_l90: err rms=1.537 initial=1.568 final=1.678 max=1.693 closed=-0.070 | settle=n/a (n=0/100) | progress=-0.010 net=-0.009 yaw_rate_ratio=0.272
  turn_r90: err rms=1.441 initial=1.575 final=1.486 max=1.575 closed=0.057 | settle=n/a (n=0/100) | progress=0.007 net=0.007 yaw_rate_ratio=0.200
  move_turn: err rms=1.463 initial=1.567 final=1.451 max=1.570 closed=0.074 | settle=n/a (n=0/100) | progress=0.034 net=0.035 yaw_rate_ratio=0.481
  move_turn_r: err rms=1.437 initial=1.576 final=1.385 max=1.576 closed=0.122 | settle=n/a (n=0/100) | progress=0.068 net=0.071 yaw_rate_ratio=0.454

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=-0.000 tripod=0.452 | closed=n/a progress=n/a net=0.000 | drift=0.032 dir_err=n/a
        fwd: n=100 surv=1.000 (fwd=1.00) | ratio=0.940 achieved=0.235 tripod=0.677 | closed=n/a progress=0.000 net=0.006 | drift=0.236 dir_err=0.029
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=0.967 achieved=0.213 tripod=0.743 | closed=n/a progress=n/a net=-0.000 | drift=0.215 dir_err=0.019
       back: n=100 surv=1.000 (back=1.00) | ratio=0.963 achieved=0.212 tripod=0.702 | closed=n/a progress=0.001 net=-0.005 | drift=0.212 dir_err=0.035
       diag: n=100 surv=1.000 (diag=1.00) | ratio=0.905 achieved=0.226 tripod=0.668 | closed=n/a progress=n/a net=0.001 | drift=0.235 dir_err=0.017
       turn: n=200 surv=1.000 (turn_l90=1.00 turn_r90=1.00) | ratio=n/a achieved=0.001 tripod=0.761 | closed=0.027 progress=0.003 net=0.003 | drift=0.051 dir_err=n/a
  move_turn: n=200 surv=0.000 (move_turn=0.00 move_turn_r=0.00) | ratio=0.735 achieved=0.162 tripod=0.416 | closed=0.099 progress=0.053 net=0.055 | drift=0.175 dir_err=0.089

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.227 action_fft_freq=2.632 joint_speed_mean=0.271 rod_speed_mean=0.029 contact_switch_hz=2.514 foot_slip=0.051 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.020 settle_six_down_s=12.000 foot_disp_end=0.072 work_per_s=231.915
        fwd: action_jitter=0.177 action_fft_freq=1.439 joint_speed_mean=0.682 rod_speed_mean=0.027 contact_switch_hz=2.083 foot_slip=0.098 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=26.951 slip_per_m=1.247 work_per_m=1020.105 cam_work_per_m=419.116
       side: action_jitter=0.123 action_fft_freq=1.150 joint_speed_mean=0.340 rod_speed_mean=0.028 contact_switch_hz=1.347 foot_slip=0.103 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=19.359 slip_per_m=1.497 work_per_m=1381.021 cam_work_per_m=228.166
       back: action_jitter=0.198 action_fft_freq=1.626 joint_speed_mean=0.621 rod_speed_mean=0.027 contact_switch_hz=1.917 foot_slip=0.089 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=27.271 slip_per_m=1.263 work_per_m=1131.137 cam_work_per_m=384.514
       diag: action_jitter=0.138 action_fft_freq=1.251 joint_speed_mean=0.592 rod_speed_mean=0.028 contact_switch_hz=1.840 foot_slip=0.105 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=24.788 slip_per_m=1.409 work_per_m=1120.238 cam_work_per_m=324.320
       turn: action_jitter=0.173 action_fft_freq=1.624 joint_speed_mean=0.424 rod_speed_mean=0.029 contact_switch_hz=1.667 foot_slip=0.067 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=567.779 slip_per_rad=23.263 work_per_rad=25764.479 cam_work_per_rad=5046.966
  move_turn: action_jitter=0.151 action_fft_freq=1.668 joint_speed_mean=0.705 rod_speed_mean=0.029 contact_switch_hz=2.290 foot_slip=0.116 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=113.269 slip_per_rad=5.710 work_per_rad=4149.532 cam_work_per_rad=2342.852

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.311 steps_per_s=3.667 gait_hz=1.099 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=11.335 support_feet=2.986 taps_per_m=15.583 swing_apex=0.121 step_rate_hz=2.667 gait_hz=0.360 clock_hz=0.357 cadence_ratio=1.008 hz_speed_slope=0.068 offset_R=0.996
       side: steps_per_m=9.633 support_feet=3.082 taps_per_m=9.370 swing_apex=0.120 step_rate_hz=2.000 gait_hz=0.311 clock_hz=0.314 cadence_ratio=0.990 hz_speed_slope=0.127 offset_R=0.991
       back: steps_per_m=11.286 support_feet=2.962 taps_per_m=15.704 swing_apex=0.129 step_rate_hz=2.333 gait_hz=0.317 clock_hz=0.314 cadence_ratio=1.010 hz_speed_slope=0.224 offset_R=0.998
       diag: steps_per_m=11.246 support_feet=3.030 taps_per_m=13.579 swing_apex=0.103 step_rate_hz=2.500 gait_hz=0.354 clock_hz=0.357 cadence_ratio=0.991 hz_speed_slope=0.029 offset_R=0.997
       turn: steps_per_rad=339.665 support_feet=3.034 taps_per_rad=204.243 swing_apex=0.129 step_rate_hz=3.083 gait_hz=0.500 clock_hz=0.500 cadence_ratio=1.001 steps_per_rad_pre=339.665 steps_per_rad_post=n/a offset_R=0.995
  move_turn: steps_per_rad=55.113 support_feet=2.885 taps_per_rad=61.007 swing_apex=0.085 step_rate_hz=3.261 gait_hz=0.439 clock_hz=0.500 cadence_ratio=0.879 hz_speed_slope=-0.137 offset_R=0.982

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
