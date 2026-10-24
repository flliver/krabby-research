```
=== crab-hex gait eval ===
scenario   : suite__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-10-21_20-59-46_seed19_X05_c6x_A15pB/model_14995.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6806261125163753  p25=0.547276905733487  p75=0.7528379549912935
tippy_tap_fraction median=0.43030615248748894
slip_ratio         median=0.24896430084274518
tracking_ratio     median=0.9762470816912856  (achieved/commanded vx, walking holds; n=700)
schedule_completion_rate=0.903
terminations={'schedule_complete': 903, 'fall': 97}
facing_error_rms   median=0.057725535057380095  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.7508777513380182 (n=599)
     stand: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.7587390114644973 (n=100)
       fwd: cmd 0.25 -> achieved 0.247 m/s | tripod median=0.5554620843450295 (n=100)
     lat_l: cmd 0.22 -> achieved 0.228 m/s | tripod median=0.7833970728997568 (n=100)
     lat_r: cmd 0.22 -> achieved 0.221 m/s | tripod median=0.7334883247469897 (n=100)
      back: cmd 0.22 -> achieved 0.216 m/s | tripod median=0.5532784765537124 (n=100)
      diag: cmd 0.25 -> achieved 0.244 m/s | tripod median=0.662179321427034 (n=100)
  turn_l90: cmd 0.00 -> achieved 0.010 m/s | tripod median=0.6948881140997323 (n=100)
  turn_r90: cmd 0.00 -> achieved -0.024 m/s | tripod median=0.7666849932059966 (n=100)
  move_turn: cmd 0.22 -> achieved 0.123 m/s | tripod median=0.47239260613917156 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.130 m/s | tripod median=0.2890573906452424 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.014 initial=0.010 final=0.015 max=0.028 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.002 yaw_rate_ratio=n/a
     stand: err rms=0.020 initial=0.016 final=0.015 max=0.044 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.056 initial=0.013 final=0.070 max=0.116 closed=n/a | settle=0.000 (n=99/100) | progress=0.001 net=0.005 yaw_rate_ratio=0.468
     lat_l: err rms=0.021 initial=0.014 final=0.016 max=0.048 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
     lat_r: err rms=0.024 initial=0.013 final=0.018 max=0.053 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
      back: err rms=0.071 initial=0.014 final=0.095 max=0.136 closed=n/a | settle=0.000 (n=99/100) | progress=0.000 net=-0.003 yaw_rate_ratio=0.309
      diag: err rms=0.038 initial=0.016 final=0.039 max=0.081 closed=n/a | settle=0.000 (n=100/100) | progress=-0.043 net=0.003 yaw_rate_ratio=0.213
  turn_l90: err rms=1.157 initial=1.578 final=0.787 max=1.579 closed=0.499 | settle=n/a (n=0/100) | progress=0.066 net=0.066 yaw_rate_ratio=0.422
  turn_r90: err rms=1.204 initial=1.564 final=0.980 max=1.566 closed=0.369 | settle=n/a (n=0/100) | progress=0.048 net=0.048 yaw_rate_ratio=0.415
  move_turn: err rms=1.112 initial=1.580 final=0.686 max=1.585 closed=0.568 | settle=n/a (n=0/100) | progress=0.074 net=0.075 yaw_rate_ratio=0.420
  move_turn_r: err rms=1.309 initial=1.563 final=1.149 max=1.566 closed=0.261 | settle=n/a (n=0/100) | progress=0.052 net=0.054 yaw_rate_ratio=0.439

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.001 tripod=0.759 | closed=n/a progress=n/a net=0.001 | drift=0.031 dir_err=n/a
        fwd: n=100 surv=1.000 (fwd=1.00) | ratio=0.986 achieved=0.247 tripod=0.555 | closed=n/a progress=0.001 net=0.005 | drift=0.246 dir_err=0.005
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=1.026 achieved=0.226 tripod=0.763 | closed=n/a progress=n/a net=0.001 | drift=0.225 dir_err=0.014
       back: n=100 surv=1.000 (back=1.00) | ratio=0.983 achieved=0.216 tripod=0.553 | closed=n/a progress=0.000 net=-0.003 | drift=0.216 dir_err=0.047
       diag: n=100 surv=1.000 (diag=1.00) | ratio=0.975 achieved=0.244 tripod=0.662 | closed=n/a progress=-0.043 net=0.003 | drift=0.244 dir_err=0.026
       turn: n=200 surv=1.000 (turn_l90=1.00 turn_r90=1.00) | ratio=n/a achieved=-0.006 tripod=0.743 | closed=0.441 progress=0.058 net=0.057 | drift=0.065 dir_err=n/a
  move_turn: n=200 surv=0.515 (move_turn=0.99 move_turn_r=0.04) | ratio=0.570 achieved=0.125 tripod=0.336 | closed=0.387 progress=0.061 net=0.062 | drift=0.156 dir_err=0.177

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.246 action_fft_freq=2.356 joint_speed_mean=0.372 rod_speed_mean=0.026 contact_switch_hz=1.750 foot_slip=0.071 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.005 settle_six_down_s=12.000 foot_disp_end=0.175 work_per_s=249.240
        fwd: action_jitter=0.231 action_fft_freq=1.767 joint_speed_mean=0.664 rod_speed_mean=0.029 contact_switch_hz=2.201 foot_slip=0.124 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=27.143 slip_per_m=1.477 work_per_m=1077.112 cam_work_per_m=507.454
       side: action_jitter=0.182 action_fft_freq=1.597 joint_speed_mean=0.355 rod_speed_mean=0.028 contact_switch_hz=1.632 foot_slip=0.089 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=22.081 slip_per_m=1.242 work_per_m=1081.400 cam_work_per_m=285.485
       back: action_jitter=0.237 action_fft_freq=1.827 joint_speed_mean=0.633 rod_speed_mean=0.029 contact_switch_hz=2.167 foot_slip=0.123 wz_active_frac=0.009 clock_restarts=0.000 swings_per_m=30.372 slip_per_m=1.670 work_per_m=1182.194 cam_work_per_m=607.985
       diag: action_jitter=0.199 action_fft_freq=1.618 joint_speed_mean=0.586 rod_speed_mean=0.029 contact_switch_hz=2.174 foot_slip=0.118 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=26.948 slip_per_m=1.475 work_per_m=1106.888 cam_work_per_m=415.032
       turn: action_jitter=0.161 action_fft_freq=1.325 joint_speed_mean=0.682 rod_speed_mean=0.028 contact_switch_hz=1.611 foot_slip=0.090 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=84.477 slip_per_rad=4.695 work_per_rad=4473.963 cam_work_per_rad=1060.566
  move_turn: action_jitter=0.183 action_fft_freq=1.541 joint_speed_mean=0.586 rod_speed_mean=0.028 contact_switch_hz=2.131 foot_slip=0.119 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=100.468 slip_per_rad=5.271 work_per_rad=4656.981 cam_work_per_rad=1829.273

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.077 steps_per_s=3.250 gait_hz=0.543 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=12.042 support_feet=2.941 taps_per_m=14.983 swing_apex=0.101 step_rate_hz=2.917 gait_hz=0.446 clock_hz=0.357 cadence_ratio=1.248 hz_speed_slope=0.291 offset_R=0.062
       side: steps_per_m=13.389 support_feet=3.100 taps_per_m=9.025 swing_apex=0.083 step_rate_hz=2.917 gait_hz=0.436 clock_hz=0.314 cadence_ratio=1.389 hz_speed_slope=8.412 offset_R=0.035
       back: steps_per_m=13.936 support_feet=2.913 taps_per_m=16.499 swing_apex=0.096 step_rate_hz=3.000 gait_hz=0.439 clock_hz=0.315 cadence_ratio=1.379 hz_speed_slope=0.674 offset_R=0.090
       diag: steps_per_m=13.349 support_feet=3.017 taps_per_m=13.201 swing_apex=0.091 step_rate_hz=3.250 gait_hz=0.494 clock_hz=0.357 cadence_ratio=1.384 hz_speed_slope=0.394 offset_R=0.077
       turn: steps_per_rad=53.186 support_feet=3.008 taps_per_rad=30.807 swing_apex=0.109 step_rate_hz=3.083 gait_hz=0.488 clock_hz=0.500 cadence_ratio=0.977 steps_per_rad_pre=53.186 steps_per_rad_post=n/a offset_R=0.070
  move_turn: steps_per_rad=50.904 support_feet=2.855 taps_per_rad=50.419 swing_apex=0.080 step_rate_hz=3.250 gait_hz=0.452 clock_hz=0.500 cadence_ratio=0.903 hz_speed_slope=0.125 offset_R=0.033

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
