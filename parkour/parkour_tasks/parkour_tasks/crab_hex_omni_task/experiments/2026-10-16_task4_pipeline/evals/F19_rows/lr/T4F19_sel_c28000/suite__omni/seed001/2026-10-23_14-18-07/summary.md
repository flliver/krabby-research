```
=== crab-hex gait eval ===
scenario   : suite__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_28000.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7060263807943752  p25=0.6669764342743396  p75=0.7571702845300521
tippy_tap_fraction median=0.42857142857142855
slip_ratio         median=0.2277823322787998
tracking_ratio     median=0.9946207469973105  (achieved/commanded vx, walking holds; n=700)
schedule_completion_rate=1.0
terminations={'schedule_complete': 1000}
facing_error_rms   median=0.06354648061337376  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.6601641854066987 (n=955)
     stand: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.7540630405422881 (n=100)
       fwd: cmd 0.25 -> achieved 0.248 m/s | tripod median=0.6811882507968084 (n=100)
     lat_l: cmd 0.22 -> achieved 0.229 m/s | tripod median=0.6603129974315918 (n=100)
     lat_r: cmd 0.22 -> achieved 0.231 m/s | tripod median=0.7582642067402592 (n=100)
      back: cmd 0.22 -> achieved 0.215 m/s | tripod median=0.6993330840908141 (n=100)
      diag: cmd 0.25 -> achieved 0.252 m/s | tripod median=0.694147283782711 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.005 m/s | tripod median=0.7950904965175931 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.006 m/s | tripod median=0.79182150667387 (n=100)
  move_turn: cmd 0.22 -> achieved 0.160 m/s | tripod median=0.6491529064524773 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.163 m/s | tripod median=0.7190402505395865 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.015 initial=0.009 final=0.017 max=0.031 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.001 yaw_rate_ratio=n/a
     stand: err rms=0.031 initial=0.019 final=0.038 max=0.063 closed=n/a | settle=0.000 (n=100/100) | progress=-0.003 net=-0.000 yaw_rate_ratio=0.496
       fwd: err rms=0.053 initial=0.015 final=0.036 max=0.115 closed=n/a | settle=0.000 (n=100/100) | progress=-0.001 net=0.003 yaw_rate_ratio=0.608
     lat_l: err rms=0.020 initial=0.017 final=0.019 max=0.046 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.003 yaw_rate_ratio=n/a
     lat_r: err rms=0.033 initial=0.017 final=0.040 max=0.071 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.000 yaw_rate_ratio=n/a
      back: err rms=0.085 initial=0.017 final=0.099 max=0.155 closed=n/a | settle=8.540 (n=99/100) | progress=-0.003 net=-0.004 yaw_rate_ratio=0.574
      diag: err rms=0.039 initial=0.017 final=0.052 max=0.088 closed=n/a | settle=0.000 (n=100/100) | progress=-0.001 net=0.003 yaw_rate_ratio=0.303
  turn_l90: err rms=0.780 initial=1.557 final=0.019 max=1.562 closed=0.988 | settle=8.260 (n=100/100) | progress=0.172 net=0.130 yaw_rate_ratio=0.667
  turn_r90: err rms=0.811 initial=1.579 final=0.023 max=1.584 closed=0.985 | settle=8.470 (n=100/100) | progress=0.170 net=0.131 yaw_rate_ratio=0.645
  move_turn: err rms=1.103 initial=1.564 final=0.627 max=1.568 closed=0.597 | settle=n/a (n=0/100) | progress=0.077 net=0.078 yaw_rate_ratio=0.404
  move_turn_r: err rms=1.007 initial=1.580 final=0.312 max=1.587 closed=0.801 | settle=11.940 (n=5/100) | progress=0.106 net=0.106 yaw_rate_ratio=0.463

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.002 tripod=0.754 | closed=n/a progress=-0.003 net=-0.000 | drift=0.035 dir_err=n/a
        fwd: n=100 surv=1.000 (fwd=1.00) | ratio=0.992 achieved=0.248 tripod=0.681 | closed=n/a progress=-0.001 net=0.003 | drift=0.248 dir_err=0.016
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=1.046 achieved=0.230 tripod=0.711 | closed=n/a progress=n/a net=0.001 | drift=0.232 dir_err=0.016
       back: n=100 surv=1.000 (back=1.00) | ratio=0.978 achieved=0.215 tripod=0.699 | closed=n/a progress=-0.003 net=-0.004 | drift=0.216 dir_err=0.034
       diag: n=100 surv=1.000 (diag=1.00) | ratio=1.009 achieved=0.252 tripod=0.694 | closed=n/a progress=-0.001 net=0.003 | drift=0.258 dir_err=0.060
       turn: n=200 surv=1.000 (turn_l90=1.00 turn_r90=1.00) | ratio=n/a achieved=0.000 tripod=0.793 | closed=0.986 progress=0.171 net=0.130 | drift=0.047 dir_err=n/a
  move_turn: n=200 surv=1.000 (move_turn=1.00 move_turn_r=1.00) | ratio=0.736 achieved=0.162 tripod=0.687 | closed=0.698 progress=0.092 net=0.092 | drift=0.176 dir_err=0.172

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.144 action_fft_freq=1.637 joint_speed_mean=0.362 rod_speed_mean=0.029 contact_switch_hz=1.632 foot_slip=0.075 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.010 settle_six_down_s=12.000 foot_disp_end=0.117 work_per_s=299.380
        fwd: action_jitter=0.105 action_fft_freq=1.062 joint_speed_mean=0.752 rod_speed_mean=0.029 contact_switch_hz=1.833 foot_slip=0.094 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=22.357 slip_per_m=1.130 work_per_m=1110.542 cam_work_per_m=411.451
       side: action_jitter=0.083 action_fft_freq=1.138 joint_speed_mean=0.383 rod_speed_mean=0.028 contact_switch_hz=1.375 foot_slip=0.074 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=18.164 slip_per_m=0.992 work_per_m=1045.139 cam_work_per_m=198.952
       back: action_jitter=0.114 action_fft_freq=1.147 joint_speed_mean=0.702 rod_speed_mean=0.029 contact_switch_hz=1.861 foot_slip=0.092 wz_active_frac=0.042 clock_restarts=0.000 swings_per_m=26.248 slip_per_m=1.265 work_per_m=1209.858 cam_work_per_m=391.582
       diag: action_jitter=0.097 action_fft_freq=1.191 joint_speed_mean=0.568 rod_speed_mean=0.029 contact_switch_hz=1.632 foot_slip=0.088 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=19.260 slip_per_m=1.029 work_per_m=1024.902 cam_work_per_m=207.446
       turn: action_jitter=0.110 action_fft_freq=1.113 joint_speed_mean=0.684 rod_speed_mean=0.029 contact_switch_hz=1.583 foot_slip=0.077 wz_active_frac=0.705 clock_restarts=0.000 swings_per_rad=36.553 slip_per_rad=1.836 work_per_rad=1896.184 cam_work_per_rad=618.488
  move_turn: action_jitter=0.093 action_fft_freq=1.039 joint_speed_mean=0.683 rod_speed_mean=0.028 contact_switch_hz=1.861 foot_slip=0.100 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=61.021 slip_per_rad=3.307 work_per_rad=3143.813 cam_work_per_rad=1080.637

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.207 steps_per_s=2.333 gait_hz=0.391 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=10.855 support_feet=2.963 taps_per_m=11.060 swing_apex=0.120 step_rate_hz=2.667 gait_hz=0.397 clock_hz=0.357 cadence_ratio=1.111 hz_speed_slope=-0.229 offset_R=0.131
       side: steps_per_m=11.165 support_feet=3.042 taps_per_m=6.943 swing_apex=0.099 step_rate_hz=2.583 gait_hz=0.403 clock_hz=0.314 cadence_ratio=1.282 hz_speed_slope=2.155 offset_R=0.040
       back: steps_per_m=13.027 support_feet=2.968 taps_per_m=13.230 swing_apex=0.109 step_rate_hz=2.750 gait_hz=0.421 clock_hz=0.317 cadence_ratio=1.316 hz_speed_slope=-0.011 offset_R=0.106
       diag: steps_per_m=12.047 support_feet=2.950 taps_per_m=7.164 swing_apex=0.093 step_rate_hz=3.083 gait_hz=0.464 clock_hz=0.357 cadence_ratio=1.299 hz_speed_slope=-0.135 offset_R=0.052
       turn: steps_per_rad=19.805 support_feet=3.118 taps_per_rad=16.281 swing_apex=0.139 step_rate_hz=2.583 gait_hz=0.439 clock_hz=0.351 cadence_ratio=1.252 steps_per_rad_pre=13.917 steps_per_rad_post=5.779 offset_R=0.019
  move_turn: steps_per_rad=29.628 support_feet=3.000 taps_per_rad=30.176 swing_apex=0.107 step_rate_hz=2.750 gait_hz=0.366 clock_hz=0.500 cadence_ratio=0.732 hz_speed_slope=0.283 offset_R=0.020

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
