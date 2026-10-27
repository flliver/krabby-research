```
=== crab-hex gait eval ===
scenario   : suite__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7187598099994064  p25=0.684629495024279  p75=0.772134949154496
tippy_tap_fraction median=0.4640269151138716
slip_ratio         median=0.24496468038056912
tracking_ratio     median=1.0026972783114787  (achieved/commanded vx, walking holds; n=700)
schedule_completion_rate=1.0
terminations={'schedule_complete': 1000}
facing_error_rms   median=0.05170053617227563  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.770376857807894 (n=975)
     stand: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.7881674914896661 (n=100)
       fwd: cmd 0.25 -> achieved 0.252 m/s | tripod median=0.6632146781309556 (n=100)
     lat_l: cmd 0.22 -> achieved 0.235 m/s | tripod median=0.6765649876808209 (n=100)
     lat_r: cmd 0.22 -> achieved 0.233 m/s | tripod median=0.7631509705371504 (n=100)
      back: cmd 0.22 -> achieved 0.217 m/s | tripod median=0.6991126335888473 (n=100)
      diag: cmd 0.25 -> achieved 0.253 m/s | tripod median=0.6943066924574209 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.004 m/s | tripod median=0.7986123252610985 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.007 m/s | tripod median=0.7940021617202169 (n=100)
  move_turn: cmd 0.22 -> achieved 0.159 m/s | tripod median=0.6650602792601248 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.166 m/s | tripod median=0.7011547164750285 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.013 initial=0.008 final=0.012 max=0.027 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.001 yaw_rate_ratio=n/a
     stand: err rms=0.025 initial=0.011 final=0.025 max=0.053 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a
       fwd: err rms=0.046 initial=0.013 final=0.040 max=0.100 closed=n/a | settle=0.000 (n=98/100) | progress=-0.001 net=0.002 yaw_rate_ratio=0.418
     lat_l: err rms=0.018 initial=0.012 final=0.013 max=0.044 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
     lat_r: err rms=0.031 initial=0.014 final=0.028 max=0.067 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
      back: err rms=0.060 initial=0.012 final=0.077 max=0.122 closed=n/a | settle=0.000 (n=100/100) | progress=-0.001 net=-0.002 yaw_rate_ratio=0.438
      diag: err rms=0.034 initial=0.014 final=0.043 max=0.077 closed=n/a | settle=0.000 (n=100/100) | progress=-0.001 net=0.000 yaw_rate_ratio=0.346
  turn_l90: err rms=0.781 initial=1.566 final=0.023 max=1.568 closed=0.985 | settle=8.120 (n=100/100) | progress=0.176 net=0.132 yaw_rate_ratio=0.660
  turn_r90: err rms=0.804 initial=1.576 final=0.019 max=1.579 closed=0.988 | settle=8.400 (n=100/100) | progress=0.171 net=0.130 yaw_rate_ratio=0.643
  move_turn: err rms=1.081 initial=1.568 final=0.592 max=1.570 closed=0.627 | settle=n/a (n=0/100) | progress=0.081 net=0.082 yaw_rate_ratio=0.420
  move_turn_r: err rms=1.021 initial=1.575 final=0.365 max=1.580 closed=0.770 | settle=11.830 (n=2/100) | progress=0.101 net=0.101 yaw_rate_ratio=0.453

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.004 tripod=0.788 | closed=n/a progress=n/a net=0.000 | drift=0.033 dir_err=n/a
        fwd: n=100 surv=1.000 (fwd=1.00) | ratio=1.009 achieved=0.252 tripod=0.663 | closed=n/a progress=-0.001 net=0.002 | drift=0.252 dir_err=0.010
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=1.062 achieved=0.234 tripod=0.716 | closed=n/a progress=n/a net=0.001 | drift=0.234 dir_err=0.021
       back: n=100 surv=1.000 (back=1.00) | ratio=0.988 achieved=0.217 tripod=0.699 | closed=n/a progress=-0.001 net=-0.002 | drift=0.218 dir_err=0.017
       diag: n=100 surv=1.000 (diag=1.00) | ratio=1.012 achieved=0.253 tripod=0.694 | closed=n/a progress=-0.001 net=0.000 | drift=0.256 dir_err=0.063
       turn: n=200 surv=1.000 (turn_l90=1.00 turn_r90=1.00) | ratio=n/a achieved=0.001 tripod=0.797 | closed=0.986 progress=0.173 net=0.131 | drift=0.046 dir_err=n/a
  move_turn: n=200 surv=1.000 (move_turn=1.00 move_turn_r=1.00) | ratio=0.742 achieved=0.163 tripod=0.688 | closed=0.685 progress=0.089 net=0.090 | drift=0.178 dir_err=0.155

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.208 action_fft_freq=1.992 joint_speed_mean=0.364 rod_speed_mean=0.029 contact_switch_hz=1.722 foot_slip=0.073 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.013 settle_six_down_s=12.000 foot_disp_end=0.090 work_per_s=288.785
        fwd: action_jitter=0.133 action_fft_freq=1.152 joint_speed_mean=0.752 rod_speed_mean=0.029 contact_switch_hz=1.938 foot_slip=0.096 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=22.923 slip_per_m=1.125 work_per_m=1092.260 cam_work_per_m=417.744
       side: action_jitter=0.107 action_fft_freq=1.231 joint_speed_mean=0.389 rod_speed_mean=0.029 contact_switch_hz=1.472 foot_slip=0.074 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=19.153 slip_per_m=0.979 work_per_m=1008.455 cam_work_per_m=190.446
       back: action_jitter=0.144 action_fft_freq=1.267 joint_speed_mean=0.702 rod_speed_mean=0.029 contact_switch_hz=1.944 foot_slip=0.091 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=26.738 slip_per_m=1.239 work_per_m=1182.020 cam_work_per_m=387.431
       diag: action_jitter=0.122 action_fft_freq=1.268 joint_speed_mean=0.584 rod_speed_mean=0.029 contact_switch_hz=1.757 foot_slip=0.088 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=20.969 slip_per_m=1.038 work_per_m=1001.568 cam_work_per_m=224.652
       turn: action_jitter=0.153 action_fft_freq=1.280 joint_speed_mean=0.680 rod_speed_mean=0.029 contact_switch_hz=1.667 foot_slip=0.074 wz_active_frac=0.693 clock_restarts=0.000 swings_per_rad=38.232 slip_per_rad=1.769 work_per_rad=1839.344 cam_work_per_rad=588.265
  move_turn: action_jitter=0.121 action_fft_freq=1.124 joint_speed_mean=0.690 rod_speed_mean=0.028 contact_switch_hz=1.972 foot_slip=0.102 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=64.260 slip_per_rad=3.388 work_per_rad=3195.881 cam_work_per_rad=1113.629

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.163 steps_per_s=2.500 gait_hz=0.427 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=10.850 support_feet=2.948 taps_per_m=12.166 swing_apex=0.119 step_rate_hz=2.750 gait_hz=0.397 clock_hz=0.357 cadence_ratio=1.111 hz_speed_slope=0.044 offset_R=0.160
       side: steps_per_m=11.036 support_feet=3.062 taps_per_m=8.237 swing_apex=0.102 step_rate_hz=2.583 gait_hz=0.418 clock_hz=0.314 cadence_ratio=1.330 hz_speed_slope=3.015 offset_R=0.034
       back: steps_per_m=12.750 support_feet=2.962 taps_per_m=14.152 swing_apex=0.110 step_rate_hz=2.750 gait_hz=0.421 clock_hz=0.314 cadence_ratio=1.340 hz_speed_slope=-0.024 offset_R=0.032
       diag: steps_per_m=11.817 support_feet=2.952 taps_per_m=8.667 swing_apex=0.095 step_rate_hz=3.000 gait_hz=0.470 clock_hz=0.357 cadence_ratio=1.316 hz_speed_slope=-0.732 offset_R=0.029
       turn: steps_per_rad=19.725 support_feet=3.112 taps_per_rad=18.034 swing_apex=0.139 step_rate_hz=2.583 gait_hz=0.433 clock_hz=0.345 cadence_ratio=1.260 steps_per_rad_pre=13.588 steps_per_rad_post=5.832 offset_R=0.039
  move_turn: steps_per_rad=30.028 support_feet=3.006 taps_per_rad=34.214 swing_apex=0.111 step_rate_hz=2.667 gait_hz=0.372 clock_hz=0.500 cadence_ratio=0.745 hz_speed_slope=-0.067 offset_R=0.026

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
