```
=== crab-hex gait eval ===
scenario   : suite__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6316339185736413  p25=0.5493093936097875  p75=0.6841693812638837
tippy_tap_fraction median=0.4189189189189189
slip_ratio         median=0.20564618864711492
tracking_ratio     median=1.025999768197682  (achieved/commanded vx, walking holds; n=700)
schedule_completion_rate=0.966
terminations={'schedule_complete': 966, 'fall': 34}
facing_error_rms   median=0.04515234150115893  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.4741073483693094 (n=919)
     stand: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7100796242932143 (n=100)
       fwd: cmd 0.25 -> achieved 0.262 m/s | tripod median=0.5583060638562964 (n=100)
     lat_l: cmd 0.22 -> achieved 0.227 m/s | tripod median=0.644356701812351 (n=100)
     lat_r: cmd 0.22 -> achieved 0.226 m/s | tripod median=0.7422113741357689 (n=100)
      back: cmd 0.22 -> achieved 0.228 m/s | tripod median=0.6463364724644836 (n=100)
      diag: cmd 0.25 -> achieved 0.261 m/s | tripod median=0.6243346560785704 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.013 m/s | tripod median=0.7173678805163737 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.010 m/s | tripod median=0.7019773451288384 (n=100)
  move_turn: cmd 0.22 -> achieved 0.184 m/s | tripod median=0.44831939788792147 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.173 m/s | tripod median=0.511305751487029 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.011 initial=0.008 final=0.012 max=0.024 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.003 yaw_rate_ratio=n/a
     stand: err rms=0.020 initial=0.011 final=0.020 max=0.048 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.035 initial=0.011 final=0.037 max=0.089 closed=n/a | settle=0.000 (n=98/100) | progress=-0.002 net=0.000 yaw_rate_ratio=0.907
     lat_l: err rms=0.019 initial=0.013 final=0.021 max=0.047 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.002 yaw_rate_ratio=n/a
     lat_r: err rms=0.024 initial=0.011 final=0.020 max=0.055 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a
      back: err rms=0.064 initial=0.012 final=0.064 max=0.130 closed=n/a | settle=0.000 (n=99/100) | progress=0.000 net=0.001 yaw_rate_ratio=0.689
      diag: err rms=0.031 initial=0.012 final=0.026 max=0.078 closed=n/a | settle=0.000 (n=99/100) | progress=-0.025 net=0.000 yaw_rate_ratio=0.575
  turn_l90: err rms=0.794 initial=1.568 final=0.019 max=1.576 closed=0.988 | settle=8.560 (n=100/100) | progress=0.166 net=0.131 yaw_rate_ratio=0.628
  turn_r90: err rms=0.798 initial=1.573 final=0.029 max=1.578 closed=0.982 | settle=8.680 (n=100/100) | progress=0.165 net=0.129 yaw_rate_ratio=0.633
  move_turn: err rms=1.231 initial=1.571 final=0.974 max=1.578 closed=0.382 | settle=n/a (n=0/100) | progress=0.050 net=0.051 yaw_rate_ratio=0.452
  move_turn_r: err rms=1.024 initial=1.573 final=0.530 max=1.578 closed=0.664 | settle=n/a (n=0/100) | progress=0.086 net=0.087 yaw_rate_ratio=0.477

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.003 tripod=0.710 | closed=n/a progress=n/a net=0.001 | drift=0.033 dir_err=n/a
        fwd: n=100 surv=0.940 (fwd=0.94) | ratio=1.048 achieved=0.262 tripod=0.558 | closed=n/a progress=-0.002 net=0.000 | drift=0.262 dir_err=0.012
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=1.032 achieved=0.227 tripod=0.685 | closed=n/a progress=n/a net=0.001 | drift=0.228 dir_err=0.018
       back: n=100 surv=1.000 (back=1.00) | ratio=1.038 achieved=0.228 tripod=0.646 | closed=n/a progress=0.000 net=0.001 | drift=0.229 dir_err=0.013
       diag: n=100 surv=0.990 (diag=0.99) | ratio=1.042 achieved=0.261 tripod=0.624 | closed=n/a progress=-0.025 net=0.000 | drift=0.262 dir_err=0.048
       turn: n=200 surv=1.000 (turn_l90=1.00 turn_r90=1.00) | ratio=n/a achieved=-0.002 tripod=0.709 | closed=0.986 progress=0.166 net=0.130 | drift=0.048 dir_err=n/a
  move_turn: n=200 surv=0.865 (move_turn=0.80 move_turn_r=0.93) | ratio=0.808 achieved=0.178 tripod=0.478 | closed=0.485 progress=0.065 net=0.066 | drift=0.196 dir_err=0.092

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.144 action_fft_freq=1.460 joint_speed_mean=0.369 rod_speed_mean=0.029 contact_switch_hz=1.333 foot_slip=0.086 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.035 settle_six_down_s=12.000 foot_disp_end=0.112 work_per_s=405.794
        fwd: action_jitter=0.107 action_fft_freq=0.982 joint_speed_mean=0.756 rod_speed_mean=0.029 contact_switch_hz=1.681 foot_slip=0.116 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=19.649 slip_per_m=1.411 work_per_m=1564.874 cam_work_per_m=622.973
       side: action_jitter=0.084 action_fft_freq=1.030 joint_speed_mean=0.387 rod_speed_mean=0.029 contact_switch_hz=1.319 foot_slip=0.096 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=17.583 slip_per_m=1.395 work_per_m=1592.546 cam_work_per_m=285.697
       back: action_jitter=0.108 action_fft_freq=1.008 joint_speed_mean=0.716 rod_speed_mean=0.029 contact_switch_hz=1.653 foot_slip=0.110 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=22.084 slip_per_m=1.591 work_per_m=1647.691 cam_work_per_m=601.947
       diag: action_jitter=0.099 action_fft_freq=1.093 joint_speed_mean=0.580 rod_speed_mean=0.029 contact_switch_hz=1.667 foot_slip=0.125 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=19.228 slip_per_m=1.531 work_per_m=1432.200 cam_work_per_m=328.965
       turn: action_jitter=0.110 action_fft_freq=1.017 joint_speed_mean=0.691 rod_speed_mean=0.029 contact_switch_hz=1.396 foot_slip=0.100 wz_active_frac=0.723 clock_restarts=0.000 swings_per_rad=32.285 slip_per_rad=2.545 work_per_rad=2680.789 cam_work_per_rad=825.026
  move_turn: action_jitter=0.101 action_fft_freq=0.997 joint_speed_mean=0.675 rod_speed_mean=0.028 contact_switch_hz=1.713 foot_slip=0.127 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=75.705 slip_per_rad=5.963 work_per_rad=6164.116 cam_work_per_rad=2889.459

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.380 steps_per_s=2.333 gait_hz=0.385 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=10.512 support_feet=3.179 taps_per_m=9.207 swing_apex=0.114 step_rate_hz=2.750 gait_hz=0.385 clock_hz=0.357 cadence_ratio=1.076 hz_speed_slope=-0.185 offset_R=0.160
       side: steps_per_m=10.972 support_feet=3.267 taps_per_m=6.178 swing_apex=0.096 step_rate_hz=2.500 gait_hz=0.388 clock_hz=0.314 cadence_ratio=1.233 hz_speed_slope=1.201 offset_R=0.102
       back: steps_per_m=12.222 support_feet=3.237 taps_per_m=9.614 swing_apex=0.105 step_rate_hz=2.750 gait_hz=0.409 clock_hz=0.314 cadence_ratio=1.288 hz_speed_slope=-0.079 offset_R=0.084
       diag: steps_per_m=11.389 support_feet=3.183 taps_per_m=7.792 swing_apex=0.092 step_rate_hz=2.917 gait_hz=0.446 clock_hz=0.357 cadence_ratio=1.248 hz_speed_slope=-0.580 offset_R=0.012
       turn: steps_per_rad=19.710 support_feet=3.303 taps_per_rad=12.097 swing_apex=0.133 step_rate_hz=2.583 gait_hz=0.403 clock_hz=0.359 cadence_ratio=1.125 steps_per_rad_pre=14.219 steps_per_rad_post=5.589 offset_R=0.092
  move_turn: steps_per_rad=41.197 support_feet=3.154 taps_per_rad=35.636 swing_apex=0.091 step_rate_hz=2.750 gait_hz=0.360 clock_hz=0.500 cadence_ratio=0.720 hz_speed_slope=-0.146 offset_R=0.038

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
