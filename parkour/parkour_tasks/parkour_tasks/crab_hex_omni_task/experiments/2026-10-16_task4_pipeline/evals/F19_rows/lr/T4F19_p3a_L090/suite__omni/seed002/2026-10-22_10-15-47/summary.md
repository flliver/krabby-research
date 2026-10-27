```
=== crab-hex gait eval ===
scenario   : suite__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_03-27-53_seed19_X07_p3a_A15pB/model_22993.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.664451005266941  p25=0.6093118436721359  p75=0.7296225817103416
tippy_tap_fraction median=0.4262295081967213
slip_ratio         median=0.20902430130514754
tracking_ratio     median=0.9952617035782809  (achieved/commanded vx, walking holds; n=700)
schedule_completion_rate=0.992
terminations={'schedule_complete': 992, 'fall': 8}
facing_error_rms   median=0.05604101587842512  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.2483797223746319 (n=312)
     stand: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.6250125719755665 (n=100)
       fwd: cmd 0.25 -> achieved 0.255 m/s | tripod median=0.6524453553760249 (n=100)
     lat_l: cmd 0.22 -> achieved 0.212 m/s | tripod median=0.6734016462918753 (n=100)
     lat_r: cmd 0.22 -> achieved 0.240 m/s | tripod median=0.7484817825969741 (n=100)
      back: cmd 0.22 -> achieved 0.230 m/s | tripod median=0.6923362257574954 (n=100)
      diag: cmd 0.25 -> achieved 0.249 m/s | tripod median=0.630981678498678 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.003 m/s | tripod median=0.7734902076480498 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.7797239510250338 (n=100)
  move_turn: cmd 0.22 -> achieved 0.176 m/s | tripod median=0.49640741369010755 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.147 m/s | tripod median=0.6512101422537415 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.010 initial=0.007 final=0.010 max=0.021 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.003 yaw_rate_ratio=n/a
     stand: err rms=0.018 initial=0.010 final=0.014 max=0.047 closed=n/a | settle=0.000 (n=100/100) | progress=0.004 net=0.001 yaw_rate_ratio=0.235
       fwd: err rms=0.049 initial=0.008 final=0.049 max=0.110 closed=n/a | settle=0.000 (n=98/100) | progress=-0.002 net=-0.000 yaw_rate_ratio=0.454
     lat_l: err rms=0.020 initial=0.010 final=0.018 max=0.049 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
     lat_r: err rms=0.014 initial=0.009 final=0.009 max=0.032 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
      back: err rms=0.071 initial=0.015 final=0.124 max=0.148 closed=n/a | settle=0.000 (n=80/100) | progress=-0.008 net=0.002 yaw_rate_ratio=0.347
      diag: err rms=0.034 initial=0.011 final=0.027 max=0.086 closed=n/a | settle=0.000 (n=96/100) | progress=-0.017 net=0.001 yaw_rate_ratio=0.459
  turn_l90: err rms=0.861 initial=1.568 final=0.133 max=1.576 closed=0.915 | settle=11.080 (n=76/100) | progress=0.131 net=0.120 yaw_rate_ratio=0.556
  turn_r90: err rms=0.865 initial=1.574 final=0.127 max=1.579 closed=0.919 | settle=10.960 (n=78/100) | progress=0.130 net=0.120 yaw_rate_ratio=0.551
  move_turn: err rms=1.187 initial=1.567 final=0.766 max=1.584 closed=0.512 | settle=n/a (n=0/100) | progress=0.067 net=0.067 yaw_rate_ratio=0.426
  move_turn_r: err rms=1.168 initial=1.578 final=0.798 max=1.580 closed=0.494 | settle=n/a (n=0/100) | progress=0.064 net=0.065 yaw_rate_ratio=0.445

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.002 tripod=0.625 | closed=n/a progress=0.004 net=0.001 | drift=0.041 dir_err=n/a
        fwd: n=100 surv=1.000 (fwd=1.00) | ratio=1.021 achieved=0.255 tripod=0.652 | closed=n/a progress=-0.002 net=-0.000 | drift=0.255 dir_err=0.032
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=1.033 achieved=0.227 tripod=0.712 | closed=n/a progress=n/a net=0.001 | drift=0.232 dir_err=0.012
       back: n=100 surv=1.000 (back=1.00) | ratio=1.047 achieved=0.230 tripod=0.692 | closed=n/a progress=-0.008 net=0.002 | drift=0.231 dir_err=0.037
       diag: n=100 surv=0.970 (diag=0.97) | ratio=0.997 achieved=0.249 tripod=0.631 | closed=n/a progress=-0.017 net=0.001 | drift=0.250 dir_err=0.024
       turn: n=200 surv=1.000 (turn_l90=1.00 turn_r90=1.00) | ratio=n/a achieved=0.001 tripod=0.776 | closed=0.918 progress=0.130 net=0.120 | drift=0.047 dir_err=n/a
  move_turn: n=200 surv=0.975 (move_turn=0.95 move_turn_r=1.00) | ratio=0.732 achieved=0.161 tripod=0.616 | closed=0.498 progress=0.065 net=0.066 | drift=0.178 dir_err=0.078

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.169 action_fft_freq=1.638 joint_speed_mean=0.343 rod_speed_mean=0.029 contact_switch_hz=1.889 foot_slip=0.085 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.003 settle_six_down_s=12.000 foot_disp_end=0.216 work_per_s=302.505
        fwd: action_jitter=0.137 action_fft_freq=1.183 joint_speed_mean=0.722 rod_speed_mean=0.029 contact_switch_hz=1.799 foot_slip=0.109 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=21.739 slip_per_m=1.290 work_per_m=1283.087 cam_work_per_m=519.699
       side: action_jitter=0.101 action_fft_freq=1.147 joint_speed_mean=0.370 rod_speed_mean=0.028 contact_switch_hz=1.278 foot_slip=0.085 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=17.383 slip_per_m=1.219 work_per_m=1278.426 cam_work_per_m=197.112
       back: action_jitter=0.135 action_fft_freq=1.174 joint_speed_mean=0.696 rod_speed_mean=0.029 contact_switch_hz=1.708 foot_slip=0.104 wz_active_frac=0.048 clock_restarts=0.000 swings_per_m=22.662 slip_per_m=1.388 work_per_m=1311.584 cam_work_per_m=541.763
       diag: action_jitter=0.121 action_fft_freq=1.203 joint_speed_mean=0.604 rod_speed_mean=0.029 contact_switch_hz=1.833 foot_slip=0.122 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=22.371 slip_per_m=1.557 work_per_m=1254.611 cam_work_per_m=403.783
       turn: action_jitter=0.132 action_fft_freq=1.108 joint_speed_mean=0.688 rod_speed_mean=0.029 contact_switch_hz=1.389 foot_slip=0.088 wz_active_frac=0.917 clock_restarts=1.000 swings_per_rad=35.073 slip_per_rad=2.348 work_per_rad=2239.387 cam_work_per_rad=755.616
  move_turn: action_jitter=0.122 action_fft_freq=1.127 joint_speed_mean=0.644 rod_speed_mean=0.028 contact_switch_hz=1.639 foot_slip=0.118 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=73.994 slip_per_rad=5.584 work_per_rad=5474.668 cam_work_per_rad=2166.966

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.218 steps_per_s=2.750 gait_hz=0.391 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=10.600 support_feet=3.020 taps_per_m=10.805 swing_apex=0.109 step_rate_hz=2.667 gait_hz=0.391 clock_hz=0.357 cadence_ratio=1.094 hz_speed_slope=0.836 offset_R=0.144
       side: steps_per_m=10.846 support_feet=3.178 taps_per_m=6.577 swing_apex=0.103 step_rate_hz=2.333 gait_hz=0.372 clock_hz=0.314 cadence_ratio=1.185 hz_speed_slope=-0.879 offset_R=0.093
       back: steps_per_m=12.012 support_feet=3.039 taps_per_m=10.490 swing_apex=0.108 step_rate_hz=2.750 gait_hz=0.397 clock_hz=0.317 cadence_ratio=1.243 hz_speed_slope=0.245 offset_R=0.082
       diag: steps_per_m=12.038 support_feet=3.156 taps_per_m=10.286 swing_apex=0.085 step_rate_hz=3.000 gait_hz=0.446 clock_hz=0.357 cadence_ratio=1.248 hz_speed_slope=-0.941 offset_R=0.051
       turn: steps_per_rad=21.238 support_feet=3.188 taps_per_rad=13.273 swing_apex=0.118 step_rate_hz=2.583 gait_hz=0.409 clock_hz=0.450 cadence_ratio=0.911 steps_per_rad_pre=18.995 steps_per_rad_post=2.091 offset_R=0.044
  move_turn: steps_per_rad=38.575 support_feet=3.055 taps_per_rad=33.480 swing_apex=0.119 step_rate_hz=2.583 gait_hz=0.363 clock_hz=0.500 cadence_ratio=0.726 hz_speed_slope=1.088 offset_R=0.018

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
