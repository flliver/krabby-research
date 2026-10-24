```
=== crab-hex gait eval ===
scenario   : suite__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7186904835171706  p25=0.6853983930228718  p75=0.7723739598441896
tippy_tap_fraction median=0.4563362381989833
slip_ratio         median=0.25441079143850687
tracking_ratio     median=1.0023248403618867  (achieved/commanded vx, walking holds; n=700)
schedule_completion_rate=1.0
terminations={'schedule_complete': 1000}
facing_error_rms   median=0.05028490308898094  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.7657500738463123 (n=967)
     stand: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7942336181860459 (n=100)
       fwd: cmd 0.25 -> achieved 0.252 m/s | tripod median=0.6698820750029388 (n=100)
     lat_l: cmd 0.22 -> achieved 0.235 m/s | tripod median=0.6771813044206618 (n=100)
     lat_r: cmd 0.22 -> achieved 0.232 m/s | tripod median=0.7632826437829439 (n=100)
      back: cmd 0.22 -> achieved 0.218 m/s | tripod median=0.7012822343766252 (n=100)
      diag: cmd 0.25 -> achieved 0.252 m/s | tripod median=0.6906026285067044 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.003 m/s | tripod median=0.7992376309019842 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.008 m/s | tripod median=0.7885439144064882 (n=100)
  move_turn: cmd 0.22 -> achieved 0.159 m/s | tripod median=0.6737819044932523 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.165 m/s | tripod median=0.7091602094724871 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.013 initial=0.008 final=0.012 max=0.027 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.001 yaw_rate_ratio=n/a
     stand: err rms=0.023 initial=0.014 final=0.021 max=0.052 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a
       fwd: err rms=0.039 initial=0.011 final=0.030 max=0.093 closed=n/a | settle=0.000 (n=100/100) | progress=-0.000 net=0.001 yaw_rate_ratio=0.436
     lat_l: err rms=0.018 initial=0.011 final=0.013 max=0.044 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
     lat_r: err rms=0.030 initial=0.011 final=0.032 max=0.069 closed=n/a | settle=0.000 (n=100/100) | progress=0.011 net=0.001 yaw_rate_ratio=0.498
      back: err rms=0.060 initial=0.013 final=0.083 max=0.122 closed=n/a | settle=0.000 (n=99/100) | progress=-0.003 net=0.001 yaw_rate_ratio=0.396
      diag: err rms=0.033 initial=0.014 final=0.035 max=0.079 closed=n/a | settle=0.000 (n=99/100) | progress=-0.001 net=0.001 yaw_rate_ratio=0.307
  turn_l90: err rms=0.780 initial=1.565 final=0.026 max=1.570 closed=0.984 | settle=8.120 (n=100/100) | progress=0.176 net=0.133 yaw_rate_ratio=0.659
  turn_r90: err rms=0.802 initial=1.571 final=0.020 max=1.574 closed=0.987 | settle=8.380 (n=100/100) | progress=0.171 net=0.130 yaw_rate_ratio=0.642
  move_turn: err rms=1.090 initial=1.570 final=0.616 max=1.573 closed=0.608 | settle=n/a (n=0/100) | progress=0.079 net=0.079 yaw_rate_ratio=0.414
  move_turn_r: err rms=1.034 initial=1.571 final=0.369 max=1.576 closed=0.766 | settle=n/a (n=0/100) | progress=0.100 net=0.101 yaw_rate_ratio=0.453

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.003 tripod=0.794 | closed=n/a progress=n/a net=0.000 | drift=0.033 dir_err=n/a
        fwd: n=100 surv=1.000 (fwd=1.00) | ratio=1.007 achieved=0.252 tripod=0.670 | closed=n/a progress=-0.000 net=0.001 | drift=0.252 dir_err=0.009
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=1.060 achieved=0.233 tripod=0.723 | closed=n/a progress=0.011 net=0.001 | drift=0.234 dir_err=0.017
       back: n=100 surv=1.000 (back=1.00) | ratio=0.992 achieved=0.218 tripod=0.701 | closed=n/a progress=-0.003 net=0.001 | drift=0.219 dir_err=0.019
       diag: n=100 surv=1.000 (diag=1.00) | ratio=1.010 achieved=0.252 tripod=0.691 | closed=n/a progress=-0.001 net=0.001 | drift=0.256 dir_err=0.063
       turn: n=200 surv=1.000 (turn_l90=1.00 turn_r90=1.00) | ratio=n/a achieved=0.002 tripod=0.795 | closed=0.985 progress=0.173 net=0.131 | drift=0.047 dir_err=n/a
  move_turn: n=200 surv=1.000 (move_turn=1.00 move_turn_r=1.00) | ratio=0.739 achieved=0.163 tripod=0.694 | closed=0.673 progress=0.088 net=0.089 | drift=0.177 dir_err=0.161

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.206 action_fft_freq=1.985 joint_speed_mean=0.365 rod_speed_mean=0.029 contact_switch_hz=1.729 foot_slip=0.073 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.012 settle_six_down_s=12.000 foot_disp_end=0.093 work_per_s=289.073
        fwd: action_jitter=0.132 action_fft_freq=1.151 joint_speed_mean=0.752 rod_speed_mean=0.029 contact_switch_hz=1.917 foot_slip=0.096 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=22.969 slip_per_m=1.123 work_per_m=1088.492 cam_work_per_m=410.066
       side: action_jitter=0.107 action_fft_freq=1.244 joint_speed_mean=0.390 rod_speed_mean=0.029 contact_switch_hz=1.472 foot_slip=0.074 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=19.043 slip_per_m=0.990 work_per_m=1011.575 cam_work_per_m=198.303
       back: action_jitter=0.142 action_fft_freq=1.257 joint_speed_mean=0.705 rod_speed_mean=0.029 contact_switch_hz=1.861 foot_slip=0.091 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=25.937 slip_per_m=1.262 work_per_m=1189.794 cam_work_per_m=390.305
       diag: action_jitter=0.121 action_fft_freq=1.268 joint_speed_mean=0.581 rod_speed_mean=0.029 contact_switch_hz=1.750 foot_slip=0.087 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=20.966 slip_per_m=1.018 work_per_m=1009.458 cam_work_per_m=224.900
       turn: action_jitter=0.151 action_fft_freq=1.271 joint_speed_mean=0.681 rod_speed_mean=0.029 contact_switch_hz=1.646 foot_slip=0.074 wz_active_frac=0.695 clock_restarts=0.000 swings_per_rad=37.677 slip_per_rad=1.757 work_per_rad=1836.444 cam_work_per_rad=598.369
  move_turn: action_jitter=0.121 action_fft_freq=1.124 joint_speed_mean=0.685 rod_speed_mean=0.028 contact_switch_hz=1.944 foot_slip=0.101 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=65.506 slip_per_rad=3.385 work_per_rad=3232.946 cam_work_per_rad=1121.594

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.161 steps_per_s=2.500 gait_hz=0.421 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=10.813 support_feet=2.952 taps_per_m=12.166 swing_apex=0.117 step_rate_hz=2.750 gait_hz=0.397 clock_hz=0.357 cadence_ratio=1.111 hz_speed_slope=-0.235 offset_R=0.069
       side: steps_per_m=11.254 support_feet=3.066 taps_per_m=7.934 swing_apex=0.098 step_rate_hz=2.625 gait_hz=0.430 clock_hz=0.314 cadence_ratio=1.369 hz_speed_slope=3.982 offset_R=0.040
       back: steps_per_m=12.715 support_feet=2.970 taps_per_m=13.094 swing_apex=0.109 step_rate_hz=2.750 gait_hz=0.421 clock_hz=0.314 cadence_ratio=1.340 hz_speed_slope=0.231 offset_R=0.083
       diag: steps_per_m=11.994 support_feet=2.944 taps_per_m=8.890 swing_apex=0.094 step_rate_hz=3.000 gait_hz=0.470 clock_hz=0.357 cadence_ratio=1.316 hz_speed_slope=-0.867 offset_R=0.133
       turn: steps_per_rad=19.642 support_feet=3.108 taps_per_rad=17.403 swing_apex=0.139 step_rate_hz=2.583 gait_hz=0.439 clock_hz=0.347 cadence_ratio=1.257 steps_per_rad_pre=13.530 steps_per_rad_post=5.779 offset_R=0.069
  move_turn: steps_per_rad=30.728 support_feet=2.998 taps_per_rad=34.163 swing_apex=0.111 step_rate_hz=2.750 gait_hz=0.372 clock_hz=0.500 cadence_ratio=0.745 hz_speed_slope=0.082 offset_R=0.059

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
