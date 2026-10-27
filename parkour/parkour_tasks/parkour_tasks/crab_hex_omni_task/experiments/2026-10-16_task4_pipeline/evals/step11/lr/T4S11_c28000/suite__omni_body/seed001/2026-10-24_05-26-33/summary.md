```
=== crab-hex gait eval ===
scenario   : suite__omni_body  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_28000.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7068330894246218  p25=0.6708386121202521  p75=0.7537629846057323
tippy_tap_fraction median=0.42857142857142855
slip_ratio         median=0.22845958113125828
tracking_ratio     median=0.9955262329647139  (achieved/commanded vx, walking holds; n=700)
schedule_completion_rate=1.0
terminations={'schedule_complete': 1000}
facing_error_rms   median=0.06191446298489956  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.6754487547316526 (n=949)
     stand: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7614354754698018 (n=100)
       fwd: cmd 0.25 -> achieved 0.248 m/s | tripod median=0.6787521198315389 (n=100)
     lat_l: cmd 0.22 -> achieved 0.230 m/s | tripod median=0.6624189391410098 (n=100)
     lat_r: cmd 0.22 -> achieved 0.230 m/s | tripod median=0.7611536909946786 (n=100)
      back: cmd 0.22 -> achieved 0.215 m/s | tripod median=0.6880460661936493 (n=100)
      diag: cmd 0.25 -> achieved 0.253 m/s | tripod median=0.6955882715823754 (n=100)
  turn_l_ceiling: cmd 0.00 -> achieved -0.003 m/s | tripod median=0.7761659300197301 (n=100)
  turn_r_ceiling: cmd 0.00 -> achieved 0.007 m/s | tripod median=0.7859397407603547 (n=100)
  move_turn_l: cmd 0.22 -> achieved 0.190 m/s | tripod median=0.6805874706466633 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.199 m/s | tripod median=0.7136984155723896 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.015 initial=0.008 final=0.016 max=0.030 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.000 yaw_rate_ratio=n/a
     stand: err rms=0.034 initial=0.020 final=0.042 max=0.065 closed=n/a | settle=0.000 (n=100/100) | progress=-0.016 net=-0.001 yaw_rate_ratio=0.785
       fwd: err rms=0.050 initial=0.017 final=0.041 max=0.122 closed=n/a | settle=0.000 (n=100/100) | progress=-0.002 net=0.003 yaw_rate_ratio=0.682
     lat_l: err rms=0.021 initial=0.015 final=0.021 max=0.049 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.002 yaw_rate_ratio=n/a
     lat_r: err rms=0.036 initial=0.016 final=0.045 max=0.077 closed=n/a | settle=0.000 (n=100/100) | progress=-0.003 net=-0.001 yaw_rate_ratio=0.292
      back: err rms=0.087 initial=0.018 final=0.099 max=0.163 closed=n/a | settle=10.960 (n=97/100) | progress=-0.001 net=-0.004 yaw_rate_ratio=0.565
      diag: err rms=0.033 initial=0.015 final=0.040 max=0.080 closed=n/a | settle=0.000 (n=100/100) | progress=0.007 net=0.003 yaw_rate_ratio=0.437
  turn_l_ceiling: err rms=0.251 initial=0.019 final=0.306 max=0.357 closed=n/a | settle=n/a (n=0/100) | progress=0.153 net=0.021 yaw_rate_ratio=0.616
  turn_r_ceiling: err rms=0.306 initial=0.016 final=0.370 max=0.453 closed=n/a | settle=n/a (n=0/100) | progress=0.147 net=-0.027 yaw_rate_ratio=0.613
  move_turn_l: err rms=0.275 initial=0.016 final=0.437 max=0.444 closed=n/a | settle=n/a (n=0/100) | progress=0.054 net=0.036 yaw_rate_ratio=0.300
  move_turn_r: err rms=0.235 initial=0.017 final=0.338 max=0.376 closed=n/a | settle=n/a (n=0/100) | progress=0.063 net=-0.025 yaw_rate_ratio=0.367

body yaw by hold (window [4 s, hold_s); medians; net = net yaw rate rad/s, ratio = net / wz_cmd SIGNED (the gated turn number), wz_ratio = mean wz / wz_cmd; ripple = std of the 1-s mean wz; drift = |mean wz| on wz = 0 holds; planar = xy drift m/s on pure turns; body = along-bearing speed / |v_cmd|, cross = mean |perp| m/s):
          settle: net=n/a ratio=n/a wz_ratio=n/a ripple=n/a | drift=n/a planar=n/a | body=n/a cross=n/a (n=0)
           stand: net=0.002 ratio=n/a wz_ratio=n/a ripple=0.017 | drift=0.003 planar=n/a | body=n/a cross=n/a (n=100)
             fwd: net=0.004 ratio=n/a wz_ratio=n/a ripple=0.044 | drift=0.007 planar=n/a | body=0.991 cross=0.026 (n=100)
           lat_l: net=-0.002 ratio=n/a wz_ratio=n/a ripple=0.016 | drift=0.002 planar=n/a | body=1.049 cross=0.026 (n=100)
           lat_r: net=0.004 ratio=n/a wz_ratio=n/a ripple=0.021 | drift=0.005 planar=n/a | body=1.051 cross=0.028 (n=100)
            back: net=0.002 ratio=n/a wz_ratio=n/a ripple=0.040 | drift=0.005 planar=n/a | body=0.973 cross=0.028 (n=100)
            diag: net=-0.005 ratio=n/a wz_ratio=n/a ripple=0.030 | drift=0.005 planar=n/a | body=1.016 cross=0.042 (n=100)
  turn_l_ceiling: net=0.161 ratio=0.960 wz_ratio=0.967 ripple=0.026 | drift=n/a planar=0.009 | body=n/a cross=n/a (n=100)
  turn_r_ceiling: net=-0.154 ratio=0.920 wz_ratio=0.922 ripple=0.045 | drift=n/a planar=0.010 | body=n/a cross=n/a (n=100)
     move_turn_l: net=0.053 ratio=0.628 wz_ratio=0.624 ripple=0.031 | drift=n/a planar=n/a | body=0.818 cross=0.027 (n=100)
     move_turn_r: net=-0.060 ratio=0.715 wz_ratio=0.713 ripple=0.062 | drift=n/a planar=n/a | body=0.874 cross=0.026 (n=100)

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.003 tripod=0.761 | closed=n/a progress=-0.016 net=-0.001 | drift=0.035 dir_err=n/a
        fwd: n=100 surv=1.000 (fwd=1.00) | ratio=0.993 achieved=0.248 tripod=0.679 | closed=n/a progress=-0.002 net=0.003 | drift=0.248 dir_err=0.018
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=1.046 achieved=0.230 tripod=0.712 | closed=n/a progress=-0.003 net=0.001 | drift=0.232 dir_err=0.017
       back: n=100 surv=1.000 (back=1.00) | ratio=0.978 achieved=0.215 tripod=0.688 | closed=n/a progress=-0.001 net=-0.004 | drift=0.216 dir_err=0.035
       diag: n=100 surv=1.000 (diag=1.00) | ratio=1.011 achieved=0.253 tripod=0.696 | closed=n/a progress=0.007 net=0.003 | drift=0.258 dir_err=0.063
       turn: n=200 surv=1.000 (turn_l_ceiling=1.00 turn_r_ceiling=1.00) | ratio=n/a achieved=0.003 tripod=0.781 | closed=n/a progress=0.150 net=-0.020 | drift=0.044 dir_err=n/a
  move_turn: n=200 surv=1.000 (move_turn_l=1.00 move_turn_r=1.00) | ratio=0.887 achieved=0.195 tripod=0.701 | closed=n/a progress=0.058 net=-0.022 | drift=0.198 dir_err=0.027

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.145 action_fft_freq=1.655 joint_speed_mean=0.362 rod_speed_mean=0.029 contact_switch_hz=1.639 foot_slip=0.075 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.010 settle_six_down_s=12.000 foot_disp_end=0.122 work_per_s=299.441
        fwd: action_jitter=0.105 action_fft_freq=1.065 joint_speed_mean=0.754 rod_speed_mean=0.029 contact_switch_hz=1.806 foot_slip=0.095 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=21.989 slip_per_m=1.136 work_per_m=1128.248 cam_work_per_m=416.088
       side: action_jitter=0.083 action_fft_freq=1.114 joint_speed_mean=0.385 rod_speed_mean=0.029 contact_switch_hz=1.431 foot_slip=0.075 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=18.893 slip_per_m=0.994 work_per_m=1050.356 cam_work_per_m=194.284
       back: action_jitter=0.110 action_fft_freq=1.132 joint_speed_mean=0.705 rod_speed_mean=0.029 contact_switch_hz=1.785 foot_slip=0.094 wz_active_frac=0.062 clock_restarts=0.000 swings_per_m=25.153 slip_per_m=1.315 work_per_m=1225.888 cam_work_per_m=415.461
       diag: action_jitter=0.097 action_fft_freq=1.191 joint_speed_mean=0.574 rod_speed_mean=0.029 contact_switch_hz=1.660 foot_slip=0.089 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=19.715 slip_per_m=1.044 work_per_m=1027.291 cam_work_per_m=207.130
       turn: action_jitter=0.121 action_fft_freq=1.213 joint_speed_mean=0.682 rod_speed_mean=0.029 contact_switch_hz=1.694 foot_slip=0.077 wz_active_frac=0.925 clock_restarts=1.000 swings_per_rad=36.273 slip_per_rad=1.734 work_per_rad=1802.000 cam_work_per_rad=614.319
  move_turn: action_jitter=0.119 action_fft_freq=1.206 joint_speed_mean=0.667 rod_speed_mean=0.029 contact_switch_hz=1.806 foot_slip=0.089 wz_active_frac=0.836 clock_restarts=0.000 swings_per_rad=105.966 slip_per_rad=5.076 work_per_rad=5405.250 cam_work_per_rad=1616.561

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.207 steps_per_s=2.333 gait_hz=0.385 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=10.690 support_feet=2.968 taps_per_m=11.097 swing_apex=0.122 step_rate_hz=2.667 gait_hz=0.397 clock_hz=0.357 cadence_ratio=1.111 hz_speed_slope=-0.261 offset_R=0.166
       side: steps_per_m=11.345 support_feet=3.047 taps_per_m=7.500 swing_apex=0.102 step_rate_hz=2.583 gait_hz=0.403 clock_hz=0.314 cadence_ratio=1.282 hz_speed_slope=3.531 offset_R=0.037
       back: steps_per_m=12.842 support_feet=2.968 taps_per_m=12.243 swing_apex=0.110 step_rate_hz=2.750 gait_hz=0.409 clock_hz=0.319 cadence_ratio=1.280 hz_speed_slope=0.836 offset_R=0.058
       diag: steps_per_m=12.199 support_feet=2.947 taps_per_m=7.476 swing_apex=0.094 step_rate_hz=3.083 gait_hz=0.464 clock_hz=0.357 cadence_ratio=1.299 hz_speed_slope=0.541 offset_R=0.054
       turn: steps_per_rad=19.853 support_feet=3.137 taps_per_rad=16.362 swing_apex=0.129 step_rate_hz=2.750 gait_hz=0.464 clock_hz=0.457 cadence_ratio=1.020 steps_per_rad_pre=0.000 steps_per_rad_post=19.853 offset_R=0.104
  move_turn: steps_per_rad=52.269 support_feet=2.923 taps_per_rad=55.369 swing_apex=0.120 step_rate_hz=2.667 gait_hz=0.366 clock_hz=0.456 cadence_ratio=0.808 hz_speed_slope=-0.826 offset_R=0.069

body yaw by category (own hold, window [4 s, hold_s); medians pooled over both hands; per-hand values are in aggregate.suite.by_category[c].by_hand):
      stand: net_yaw_rate=0.002 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.017 heading_drift=0.003 planar_drift=n/a body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
        fwd: net_yaw_rate=0.004 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.044 heading_drift=0.007 planar_drift=n/a body_ratio=0.991 cross_body=0.026 adapter_clipped_frac=n/a
       side: net_yaw_rate=0.000 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.018 heading_drift=0.004 planar_drift=n/a body_ratio=1.050 cross_body=0.027 adapter_clipped_frac=n/a
       back: net_yaw_rate=0.002 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.040 heading_drift=0.005 planar_drift=n/a body_ratio=0.973 cross_body=0.028 adapter_clipped_frac=n/a
       diag: net_yaw_rate=-0.005 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.030 heading_drift=0.005 planar_drift=n/a body_ratio=1.016 cross_body=0.042 adapter_clipped_frac=n/a
       turn: net_yaw_rate=0.004 net_yaw_ratio=0.935 wz_ratio=0.940 yaw_ripple=0.034 heading_drift=n/a planar_drift=0.010 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  move_turn: net_yaw_rate=-0.004 net_yaw_ratio=0.669 wz_ratio=0.666 yaw_ripple=0.039 heading_drift=n/a planar_drift=n/a body_ratio=0.851 cross_body=0.027 adapter_clipped_frac=n/a

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
