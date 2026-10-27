```
=== crab-hex gait eval ===
scenario   : suite__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-10-21_19-09-27_seed19_X03_c5_A15pB/model_8997.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6846474585221494  p25=0.5912179721713051  p75=0.7379521938607552
tippy_tap_fraction median=0.47344924812030076
slip_ratio         median=0.27211257338735184
tracking_ratio     median=0.9081984244673547  (achieved/commanded vx, walking holds; n=700)
schedule_completion_rate=0.976
terminations={'schedule_complete': 976, 'fall': 24}
facing_error_rms   median=0.03227453733736599  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.4100495460092853 (n=855)
     stand: cmd 0.00 -> achieved 0.000 m/s | tripod median=0.39010822565543773 (n=100)
       fwd: cmd 0.25 -> achieved 0.226 m/s | tripod median=0.7033705441476591 (n=100)
     lat_l: cmd 0.22 -> achieved 0.220 m/s | tripod median=0.810276167355205 (n=100)
     lat_r: cmd 0.22 -> achieved 0.218 m/s | tripod median=0.7821813533611575 (n=100)
      back: cmd 0.22 -> achieved 0.195 m/s | tripod median=0.6943934261073448 (n=100)
      diag: cmd 0.25 -> achieved 0.236 m/s | tripod median=0.7674739078735524 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.009 m/s | tripod median=0.6972585583064768 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.024 m/s | tripod median=0.7631400480704875 (n=100)
  move_turn: cmd 0.22 -> achieved 0.119 m/s | tripod median=0.4003308705944866 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.162 m/s | tripod median=0.6029718475890353 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.006 initial=0.005 final=0.006 max=0.011 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.000 yaw_rate_ratio=n/a
     stand: err rms=0.009 initial=0.006 final=0.012 max=0.020 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.000 yaw_rate_ratio=n/a
       fwd: err rms=0.035 initial=0.006 final=0.047 max=0.077 closed=n/a | settle=0.000 (n=100/100) | progress=-0.007 net=-0.002 yaw_rate_ratio=0.528
     lat_l: err rms=0.019 initial=0.007 final=0.020 max=0.042 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
     lat_r: err rms=0.015 initial=0.006 final=0.018 max=0.038 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a
      back: err rms=0.031 initial=0.006 final=0.023 max=0.078 closed=n/a | settle=0.000 (n=99/100) | progress=-0.001 net=-0.000 yaw_rate_ratio=0.409
      diag: err rms=0.026 initial=0.006 final=0.023 max=0.061 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
  turn_l90: err rms=1.295 initial=1.569 final=1.148 max=1.570 closed=0.269 | settle=n/a (n=0/100) | progress=0.035 net=0.035 yaw_rate_ratio=0.195
  turn_r90: err rms=1.314 initial=1.572 final=1.183 max=1.572 closed=0.248 | settle=n/a (n=0/100) | progress=0.033 net=0.032 yaw_rate_ratio=0.187
  move_turn: err rms=1.507 initial=1.570 final=1.612 max=1.642 closed=-0.027 | settle=n/a (n=0/100) | progress=-0.004 net=-0.004 yaw_rate_ratio=0.345
  move_turn_r: err rms=1.293 initial=1.572 final=1.182 max=1.574 closed=0.251 | settle=n/a (n=0/100) | progress=0.033 net=0.033 yaw_rate_ratio=0.365

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.000 tripod=0.390 | closed=n/a progress=n/a net=-0.000 | drift=0.031 dir_err=n/a
        fwd: n=100 surv=1.000 (fwd=1.00) | ratio=0.903 achieved=0.226 tripod=0.703 | closed=n/a progress=-0.007 net=-0.002 | drift=0.226 dir_err=0.013
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=0.995 achieved=0.219 tripod=0.795 | closed=n/a progress=n/a net=0.000 | drift=0.220 dir_err=0.016
       back: n=100 surv=0.990 (back=0.99) | ratio=0.889 achieved=0.195 tripod=0.694 | closed=n/a progress=-0.001 net=-0.000 | drift=0.196 dir_err=0.015
       diag: n=100 surv=1.000 (diag=1.00) | ratio=0.942 achieved=0.236 tripod=0.767 | closed=n/a progress=n/a net=0.001 | drift=0.236 dir_err=0.038
       turn: n=200 surv=1.000 (turn_l90=1.00 turn_r90=1.00) | ratio=n/a achieved=0.007 tripod=0.728 | closed=0.258 progress=0.034 net=0.034 | drift=0.052 dir_err=n/a
  move_turn: n=200 surv=0.885 (move_turn=0.77 move_turn_r=1.00) | ratio=0.659 achieved=0.145 tripod=0.516 | closed=0.091 progress=0.013 net=0.013 | drift=0.161 dir_err=0.045

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.263 action_fft_freq=2.956 joint_speed_mean=0.260 rod_speed_mean=0.029 contact_switch_hz=2.708 foot_slip=0.053 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.008 settle_six_down_s=12.000 foot_disp_end=0.087 work_per_s=227.272
        fwd: action_jitter=0.229 action_fft_freq=1.817 joint_speed_mean=0.631 rod_speed_mean=0.028 contact_switch_hz=1.806 foot_slip=0.101 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=24.332 slip_per_m=1.343 work_per_m=1094.933 cam_work_per_m=368.143
       side: action_jitter=0.130 action_fft_freq=1.129 joint_speed_mean=0.327 rod_speed_mean=0.028 contact_switch_hz=1.250 foot_slip=0.109 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=17.374 slip_per_m=1.561 work_per_m=1411.032 cam_work_per_m=314.466
       back: action_jitter=0.235 action_fft_freq=1.958 joint_speed_mean=0.574 rod_speed_mean=0.028 contact_switch_hz=1.924 foot_slip=0.103 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=30.288 slip_per_m=1.602 work_per_m=1349.228 cam_work_per_m=373.709
       diag: action_jitter=0.169 action_fft_freq=1.344 joint_speed_mean=0.524 rod_speed_mean=0.028 contact_switch_hz=1.611 foot_slip=0.102 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=20.891 slip_per_m=1.327 work_per_m=1158.950 cam_work_per_m=282.417
       turn: action_jitter=0.195 action_fft_freq=1.808 joint_speed_mean=0.447 rod_speed_mean=0.029 contact_switch_hz=1.549 foot_slip=0.077 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=138.389 slip_per_rad=6.969 work_per_rad=7397.910 cam_work_per_rad=772.733
  move_turn: action_jitter=0.180 action_fft_freq=1.537 joint_speed_mean=0.658 rod_speed_mean=0.028 contact_switch_hz=2.458 foot_slip=0.135 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=288.915 slip_per_rad=16.186 work_per_rad=11796.486 cam_work_per_rad=4164.869

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.257 steps_per_s=3.500 gait_hz=1.114 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=10.921 support_feet=2.995 taps_per_m=13.234 swing_apex=0.111 step_rate_hz=2.417 gait_hz=0.360 clock_hz=0.357 cadence_ratio=1.008 hz_speed_slope=0.138 offset_R=0.999
       side: steps_per_m=9.133 support_feet=3.077 taps_per_m=7.994 swing_apex=0.108 step_rate_hz=2.000 gait_hz=0.311 clock_hz=0.314 cadence_ratio=0.990 hz_speed_slope=0.135 offset_R=0.997
       back: steps_per_m=12.710 support_feet=3.020 taps_per_m=17.190 swing_apex=0.108 step_rate_hz=2.417 gait_hz=0.317 clock_hz=0.314 cadence_ratio=1.010 hz_speed_slope=-0.362 offset_R=0.997
       diag: steps_per_m=9.937 support_feet=3.045 taps_per_m=11.017 swing_apex=0.095 step_rate_hz=2.333 gait_hz=0.354 clock_hz=0.357 cadence_ratio=0.991 hz_speed_slope=0.098 offset_R=0.998
       turn: steps_per_rad=89.916 support_feet=3.030 taps_per_rad=46.959 swing_apex=0.116 step_rate_hz=3.000 gait_hz=0.500 clock_hz=0.500 cadence_ratio=1.001 steps_per_rad_pre=89.916 steps_per_rad_post=n/a offset_R=0.997
  move_turn: steps_per_rad=128.602 support_feet=2.949 taps_per_rad=154.975 swing_apex=0.072 step_rate_hz=3.500 gait_hz=0.494 clock_hz=0.500 cadence_ratio=0.989 hz_speed_slope=0.020 offset_R=0.998

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
