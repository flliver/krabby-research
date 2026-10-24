```
=== crab-hex gait eval ===
scenario   : suite__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6935825828495005  p25=0.6529336854885779  p75=0.7506908033632365
tippy_tap_fraction median=0.4142857142857143
slip_ratio         median=0.20871382482939388
tracking_ratio     median=1.0284420298582364  (achieved/commanded vx, walking holds; n=700)
schedule_completion_rate=1.0
terminations={'schedule_complete': 1000}
facing_error_rms   median=0.04315811035587254  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.005 m/s | tripod median=0.6879659424358027 (n=982)
     stand: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7573967990213535 (n=100)
       fwd: cmd 0.25 -> achieved 0.259 m/s | tripod median=0.6756479691047711 (n=100)
     lat_l: cmd 0.22 -> achieved 0.235 m/s | tripod median=0.66833189354745 (n=100)
     lat_r: cmd 0.22 -> achieved 0.233 m/s | tripod median=0.7638105324943978 (n=100)
      back: cmd 0.22 -> achieved 0.224 m/s | tripod median=0.7004098526187092 (n=100)
      diag: cmd 0.25 -> achieved 0.259 m/s | tripod median=0.6728536431366966 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.008 m/s | tripod median=0.7723454624958221 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.009 m/s | tripod median=0.7675676773012112 (n=100)
  move_turn: cmd 0.22 -> achieved 0.169 m/s | tripod median=0.5928411075600313 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.173 m/s | tripod median=0.6439478332573032 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.012 initial=0.007 final=0.012 max=0.027 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.002 yaw_rate_ratio=n/a
     stand: err rms=0.023 initial=0.014 final=0.022 max=0.052 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.035 initial=0.011 final=0.040 max=0.086 closed=n/a | settle=0.000 (n=100/100) | progress=-0.000 net=0.002 yaw_rate_ratio=0.593
     lat_l: err rms=0.018 initial=0.010 final=0.014 max=0.044 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
     lat_r: err rms=0.027 initial=0.014 final=0.034 max=0.063 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
      back: err rms=0.046 initial=0.011 final=0.068 max=0.104 closed=n/a | settle=0.000 (n=100/100) | progress=-0.002 net=-0.001 yaw_rate_ratio=0.436
      diag: err rms=0.033 initial=0.012 final=0.037 max=0.078 closed=n/a | settle=0.000 (n=100/100) | progress=-0.001 net=0.002 yaw_rate_ratio=0.349
  turn_l90: err rms=0.788 initial=1.570 final=0.017 max=1.575 closed=0.989 | settle=8.320 (n=100/100) | progress=0.172 net=0.132 yaw_rate_ratio=0.643
  turn_r90: err rms=0.791 initial=1.568 final=0.025 max=1.578 closed=0.984 | settle=8.440 (n=100/100) | progress=0.170 net=0.130 yaw_rate_ratio=0.643
  move_turn: err rms=1.123 initial=1.567 final=0.723 max=1.573 closed=0.538 | settle=n/a (n=0/100) | progress=0.069 net=0.070 yaw_rate_ratio=0.441
  move_turn_r: err rms=1.013 initial=1.573 final=0.388 max=1.578 closed=0.755 | settle=n/a (n=0/100) | progress=0.098 net=0.099 yaw_rate_ratio=0.473

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.003 tripod=0.757 | closed=n/a progress=n/a net=0.001 | drift=0.030 dir_err=n/a
        fwd: n=100 surv=1.000 (fwd=1.00) | ratio=1.034 achieved=0.259 tripod=0.676 | closed=n/a progress=-0.000 net=0.002 | drift=0.258 dir_err=0.009
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=1.065 achieved=0.234 tripod=0.716 | closed=n/a progress=n/a net=0.001 | drift=0.235 dir_err=0.019
       back: n=100 surv=1.000 (back=1.00) | ratio=1.020 achieved=0.224 tripod=0.700 | closed=n/a progress=-0.002 net=-0.001 | drift=0.225 dir_err=0.008
       diag: n=100 surv=1.000 (diag=1.00) | ratio=1.035 achieved=0.259 tripod=0.673 | closed=n/a progress=-0.001 net=0.002 | drift=0.261 dir_err=0.050
       turn: n=200 surv=1.000 (turn_l90=1.00 turn_r90=1.00) | ratio=n/a achieved=0.001 tripod=0.769 | closed=0.986 progress=0.171 net=0.131 | drift=0.045 dir_err=n/a
  move_turn: n=200 surv=1.000 (move_turn=1.00 move_turn_r=1.00) | ratio=0.778 achieved=0.171 tripod=0.619 | closed=0.638 progress=0.082 net=0.083 | drift=0.185 dir_err=0.107

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.170 action_fft_freq=1.652 joint_speed_mean=0.366 rod_speed_mean=0.029 contact_switch_hz=1.340 foot_slip=0.078 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.021 settle_six_down_s=12.000 foot_disp_end=0.094 work_per_s=343.641
        fwd: action_jitter=0.118 action_fft_freq=1.036 joint_speed_mean=0.759 rod_speed_mean=0.029 contact_switch_hz=1.722 foot_slip=0.103 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=20.047 slip_per_m=1.226 work_per_m=1315.147 cam_work_per_m=491.314
       side: action_jitter=0.091 action_fft_freq=1.066 joint_speed_mean=0.388 rod_speed_mean=0.029 contact_switch_hz=1.250 foot_slip=0.083 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=16.026 slip_per_m=1.116 work_per_m=1260.945 cam_work_per_m=223.076
       back: action_jitter=0.121 action_fft_freq=1.100 joint_speed_mean=0.709 rod_speed_mean=0.029 contact_switch_hz=1.639 foot_slip=0.103 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=22.289 slip_per_m=1.451 work_per_m=1428.641 cam_work_per_m=473.684
       diag: action_jitter=0.106 action_fft_freq=1.141 joint_speed_mean=0.576 rod_speed_mean=0.029 contact_switch_hz=1.639 foot_slip=0.099 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=19.171 slip_per_m=1.163 work_per_m=1194.269 cam_work_per_m=247.911
       turn: action_jitter=0.126 action_fft_freq=1.094 joint_speed_mean=0.687 rod_speed_mean=0.029 contact_switch_hz=1.312 foot_slip=0.081 wz_active_frac=0.703 clock_restarts=0.000 swings_per_rad=30.018 slip_per_rad=2.005 work_per_rad=2225.455 cam_work_per_rad=700.332
  move_turn: action_jitter=0.108 action_fft_freq=1.049 joint_speed_mean=0.686 rod_speed_mean=0.028 contact_switch_hz=1.778 foot_slip=0.110 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=61.131 slip_per_rad=4.127 work_per_rad=4180.198 cam_work_per_rad=1607.905

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.267 steps_per_s=2.417 gait_hz=0.409 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=9.935 support_feet=3.050 taps_per_m=9.864 swing_apex=0.121 step_rate_hz=2.583 gait_hz=0.391 clock_hz=0.357 cadence_ratio=1.094 hz_speed_slope=-0.230 offset_R=0.108
       side: steps_per_m=10.496 support_feet=3.141 taps_per_m=5.419 swing_apex=0.101 step_rate_hz=2.500 gait_hz=0.397 clock_hz=0.314 cadence_ratio=1.262 hz_speed_slope=5.324 offset_R=0.089
       back: steps_per_m=12.073 support_feet=3.098 taps_per_m=10.405 swing_apex=0.109 step_rate_hz=2.667 gait_hz=0.415 clock_hz=0.314 cadence_ratio=1.321 hz_speed_slope=-0.234 offset_R=0.099
       diag: steps_per_m=11.325 support_feet=3.033 taps_per_m=7.839 swing_apex=0.093 step_rate_hz=2.917 gait_hz=0.464 clock_hz=0.357 cadence_ratio=1.295 hz_speed_slope=0.153 offset_R=0.137
       turn: steps_per_rad=19.139 support_feet=3.207 taps_per_rad=10.973 swing_apex=0.137 step_rate_hz=2.500 gait_hz=0.421 clock_hz=0.349 cadence_ratio=1.206 steps_per_rad_pre=13.419 steps_per_rad_post=5.719 offset_R=0.024
  move_turn: steps_per_rad=30.757 support_feet=3.068 taps_per_rad=29.878 swing_apex=0.109 step_rate_hz=2.583 gait_hz=0.366 clock_hz=0.500 cadence_ratio=0.732 hz_speed_slope=-0.258 offset_R=0.025

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
