```
=== crab-hex gait eval ===
scenario   : suite__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_03-27-53_seed19_X07_p3a_A15pB/model_22993.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.71090252602462  p25=0.6643344460411709  p75=0.7600274880278738
tippy_tap_fraction median=0.42857142857142855
slip_ratio         median=0.21929322869919124
tracking_ratio     median=0.9811348080671072  (achieved/commanded vx, walking holds; n=700)
schedule_completion_rate=0.999
terminations={'schedule_complete': 999, 'fall': 1}
facing_error_rms   median=0.053876406667276056  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.25954991213183654 (n=258)
     stand: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.772935346569396 (n=100)
       fwd: cmd 0.25 -> achieved 0.253 m/s | tripod median=0.707450105016216 (n=100)
     lat_l: cmd 0.22 -> achieved 0.211 m/s | tripod median=0.7063048629688424 (n=100)
     lat_r: cmd 0.22 -> achieved 0.243 m/s | tripod median=0.7560226462751463 (n=100)
      back: cmd 0.22 -> achieved 0.228 m/s | tripod median=0.7314446459942228 (n=100)
      diag: cmd 0.25 -> achieved 0.245 m/s | tripod median=0.6729193629049852 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.003 m/s | tripod median=0.7982135548476084 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.8003048599526101 (n=100)
  move_turn: cmd 0.22 -> achieved 0.170 m/s | tripod median=0.602707860423132 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.140 m/s | tripod median=0.6735133872477534 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.010 initial=0.007 final=0.011 max=0.021 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.003 yaw_rate_ratio=n/a
     stand: err rms=0.020 initial=0.010 final=0.019 max=0.055 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.048 initial=0.008 final=0.067 max=0.107 closed=n/a | settle=0.000 (n=95/100) | progress=-0.000 net=-0.002 yaw_rate_ratio=0.330
     lat_l: err rms=0.021 initial=0.011 final=0.017 max=0.049 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.002 yaw_rate_ratio=n/a
     lat_r: err rms=0.015 initial=0.011 final=0.010 max=0.034 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
      back: err rms=0.075 initial=0.014 final=0.125 max=0.149 closed=n/a | settle=0.000 (n=86/100) | progress=-0.003 net=0.007 yaw_rate_ratio=0.311
      diag: err rms=0.032 initial=0.010 final=0.019 max=0.078 closed=n/a | settle=0.000 (n=98/100) | progress=-0.009 net=0.001 yaw_rate_ratio=0.318
  turn_l90: err rms=0.842 initial=1.568 final=0.123 max=1.575 closed=0.922 | settle=10.420 (n=87/100) | progress=0.138 net=0.121 yaw_rate_ratio=0.578
  turn_r90: err rms=0.847 initial=1.574 final=0.122 max=1.580 closed=0.922 | settle=10.580 (n=84/100) | progress=0.136 net=0.121 yaw_rate_ratio=0.569
  move_turn: err rms=1.132 initial=1.565 final=0.589 max=1.581 closed=0.625 | settle=n/a (n=0/100) | progress=0.082 net=0.082 yaw_rate_ratio=0.398
  move_turn_r: err rms=1.175 initial=1.575 final=0.803 max=1.578 closed=0.490 | settle=n/a (n=0/100) | progress=0.064 net=0.065 yaw_rate_ratio=0.438

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.002 tripod=0.773 | closed=n/a progress=n/a net=0.001 | drift=0.038 dir_err=n/a
        fwd: n=100 surv=1.000 (fwd=1.00) | ratio=1.010 achieved=0.253 tripod=0.707 | closed=n/a progress=-0.000 net=-0.002 | drift=0.254 dir_err=0.033
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=1.023 achieved=0.225 tripod=0.734 | closed=n/a progress=n/a net=0.001 | drift=0.226 dir_err=0.013
       back: n=100 surv=1.000 (back=1.00) | ratio=1.038 achieved=0.228 tripod=0.731 | closed=n/a progress=-0.003 net=0.007 | drift=0.229 dir_err=0.046
       diag: n=100 surv=0.990 (diag=0.99) | ratio=0.979 achieved=0.245 tripod=0.673 | closed=n/a progress=-0.009 net=0.001 | drift=0.245 dir_err=0.017
       turn: n=200 surv=1.000 (turn_l90=1.00 turn_r90=1.00) | ratio=n/a achieved=0.001 tripod=0.800 | closed=0.922 progress=0.138 net=0.121 | drift=0.048 dir_err=n/a
  move_turn: n=200 surv=1.000 (move_turn=1.00 move_turn_r=1.00) | ratio=0.712 achieved=0.157 tripod=0.656 | closed=0.544 progress=0.071 net=0.072 | drift=0.173 dir_err=0.112

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.171 action_fft_freq=1.630 joint_speed_mean=0.337 rod_speed_mean=0.029 contact_switch_hz=1.438 foot_slip=0.075 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.003 settle_six_down_s=12.000 foot_disp_end=0.210 work_per_s=274.290
        fwd: action_jitter=0.147 action_fft_freq=1.245 joint_speed_mean=0.718 rod_speed_mean=0.029 contact_switch_hz=1.792 foot_slip=0.101 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=21.474 slip_per_m=1.204 work_per_m=1161.710 cam_work_per_m=449.266
       side: action_jitter=0.113 action_fft_freq=1.235 joint_speed_mean=0.363 rod_speed_mean=0.028 contact_switch_hz=1.333 foot_slip=0.075 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=17.879 slip_per_m=1.084 work_per_m=1142.175 cam_work_per_m=177.850
       back: action_jitter=0.145 action_fft_freq=1.236 joint_speed_mean=0.690 rod_speed_mean=0.029 contact_switch_hz=1.785 foot_slip=0.094 wz_active_frac=0.057 clock_restarts=0.000 swings_per_m=23.718 slip_per_m=1.231 work_per_m=1175.688 cam_work_per_m=476.724
       diag: action_jitter=0.128 action_fft_freq=1.262 joint_speed_mean=0.601 rod_speed_mean=0.029 contact_switch_hz=1.833 foot_slip=0.097 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=22.799 slip_per_m=1.241 work_per_m=1085.723 cam_work_per_m=328.395
       turn: action_jitter=0.140 action_fft_freq=1.150 joint_speed_mean=0.686 rod_speed_mean=0.029 contact_switch_hz=1.451 foot_slip=0.082 wz_active_frac=0.873 clock_restarts=1.000 swings_per_rad=36.139 slip_per_rad=2.129 work_per_rad=2059.194 cam_work_per_rad=666.451
  move_turn: action_jitter=0.129 action_fft_freq=1.177 joint_speed_mean=0.641 rod_speed_mean=0.028 contact_switch_hz=1.653 foot_slip=0.112 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=68.597 slip_per_rad=4.760 work_per_rad=4627.548 cam_work_per_rad=1667.878

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.119 steps_per_s=2.500 gait_hz=0.381 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=10.710 support_feet=2.992 taps_per_m=10.629 swing_apex=0.111 step_rate_hz=2.667 gait_hz=0.397 clock_hz=0.357 cadence_ratio=1.111 hz_speed_slope=0.988 offset_R=0.201
       side: steps_per_m=10.979 support_feet=3.131 taps_per_m=6.732 swing_apex=0.102 step_rate_hz=2.417 gait_hz=0.385 clock_hz=0.314 cadence_ratio=1.223 hz_speed_slope=-0.940 offset_R=0.030
       back: steps_per_m=11.935 support_feet=2.992 taps_per_m=11.426 swing_apex=0.113 step_rate_hz=2.708 gait_hz=0.403 clock_hz=0.317 cadence_ratio=1.263 hz_speed_slope=0.387 offset_R=0.100
       diag: steps_per_m=12.476 support_feet=3.072 taps_per_m=10.194 swing_apex=0.090 step_rate_hz=3.000 gait_hz=0.446 clock_hz=0.357 cadence_ratio=1.248 hz_speed_slope=-0.448 offset_R=0.069
       turn: steps_per_rad=21.144 support_feet=3.132 taps_per_rad=14.478 swing_apex=0.121 step_rate_hz=2.583 gait_hz=0.415 clock_hz=0.428 cadence_ratio=0.970 steps_per_rad_pre=18.386 steps_per_rad_post=2.818 offset_R=0.078
  move_turn: steps_per_rad=35.927 support_feet=3.014 taps_per_rad=31.319 swing_apex=0.124 step_rate_hz=2.583 gait_hz=0.366 clock_hz=0.500 cadence_ratio=0.732 hz_speed_slope=0.774 offset_R=0.040

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
