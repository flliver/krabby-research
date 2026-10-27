```
=== crab-hex gait eval ===
scenario   : suite__omni_body  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_28000.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7098750278716236  p25=0.6739563913785374  p75=0.7502293988636467
tippy_tap_fraction median=0.4294019933554817
slip_ratio         median=0.22770072581521594
tracking_ratio     median=0.9941894570021894  (achieved/commanded vx, walking holds; n=700)
schedule_completion_rate=1.0
terminations={'schedule_complete': 1000}
facing_error_rms   median=0.0649966867663053  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.6701019547772873 (n=925)
     stand: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.7606510172790535 (n=100)
       fwd: cmd 0.25 -> achieved 0.248 m/s | tripod median=0.6801099122212735 (n=100)
     lat_l: cmd 0.22 -> achieved 0.230 m/s | tripod median=0.6613860564304518 (n=100)
     lat_r: cmd 0.22 -> achieved 0.229 m/s | tripod median=0.7574001668925892 (n=100)
      back: cmd 0.22 -> achieved 0.215 m/s | tripod median=0.6916210726640091 (n=100)
      diag: cmd 0.25 -> achieved 0.253 m/s | tripod median=0.6928467877819535 (n=100)
  turn_l_ceiling: cmd 0.00 -> achieved -0.001 m/s | tripod median=0.7767940636660711 (n=100)
  turn_r_ceiling: cmd 0.00 -> achieved 0.007 m/s | tripod median=0.7941355984055583 (n=100)
  move_turn_l: cmd 0.22 -> achieved 0.192 m/s | tripod median=0.6887857141522129 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.200 m/s | tripod median=0.7178690522302181 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.014 initial=0.008 final=0.015 max=0.030 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.000 yaw_rate_ratio=n/a
     stand: err rms=0.028 initial=0.017 final=0.037 max=0.061 closed=n/a | settle=0.000 (n=100/100) | progress=-0.018 net=-0.000 yaw_rate_ratio=0.371
       fwd: err rms=0.053 initial=0.015 final=0.052 max=0.127 closed=n/a | settle=0.000 (n=100/100) | progress=-0.002 net=0.004 yaw_rate_ratio=0.675
     lat_l: err rms=0.019 initial=0.013 final=0.019 max=0.047 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.002 yaw_rate_ratio=n/a
     lat_r: err rms=0.033 initial=0.015 final=0.044 max=0.070 closed=n/a | settle=0.000 (n=100/100) | progress=0.001 net=0.001 yaw_rate_ratio=0.240
      back: err rms=0.088 initial=0.013 final=0.100 max=0.176 closed=n/a | settle=11.020 (n=98/100) | progress=-0.001 net=-0.004 yaw_rate_ratio=0.592
      diag: err rms=0.032 initial=0.014 final=0.036 max=0.078 closed=n/a | settle=0.000 (n=100/100) | progress=0.003 net=0.003 yaw_rate_ratio=0.336
  turn_l_ceiling: err rms=0.246 initial=0.017 final=0.281 max=0.344 closed=n/a | settle=n/a (n=0/100) | progress=0.156 net=0.020 yaw_rate_ratio=0.623
  turn_r_ceiling: err rms=0.295 initial=0.016 final=0.377 max=0.437 closed=n/a | settle=n/a (n=0/100) | progress=0.147 net=-0.025 yaw_rate_ratio=0.613
  move_turn_l: err rms=0.279 initial=0.015 final=0.440 max=0.446 closed=n/a | settle=n/a (n=0/100) | progress=0.054 net=0.034 yaw_rate_ratio=0.300
  move_turn_r: err rms=0.236 initial=0.019 final=0.344 max=0.376 closed=n/a | settle=n/a (n=0/100) | progress=0.062 net=-0.026 yaw_rate_ratio=0.366

body yaw by hold (window [4 s, hold_s); medians; net = net yaw rate rad/s, ratio = net / wz_cmd SIGNED (the gated turn number), wz_ratio = mean wz / wz_cmd; ripple = std of the 1-s mean wz; drift = |mean wz| on wz = 0 holds; planar = xy drift m/s on pure turns; body = along-bearing speed / |v_cmd|, cross = mean |perp| m/s):
          settle: net=n/a ratio=n/a wz_ratio=n/a ripple=n/a | drift=n/a planar=n/a | body=n/a cross=n/a (n=0)
           stand: net=0.002 ratio=n/a wz_ratio=n/a ripple=0.015 | drift=0.003 planar=n/a | body=n/a cross=n/a (n=100)
             fwd: net=-0.000 ratio=n/a wz_ratio=n/a ripple=0.043 | drift=0.007 planar=n/a | body=0.988 cross=0.026 (n=100)
           lat_l: net=-0.002 ratio=n/a wz_ratio=n/a ripple=0.014 | drift=0.003 planar=n/a | body=1.045 cross=0.026 (n=100)
           lat_r: net=0.004 ratio=n/a wz_ratio=n/a ripple=0.021 | drift=0.005 planar=n/a | body=1.048 cross=0.028 (n=100)
            back: net=0.002 ratio=n/a wz_ratio=n/a ripple=0.040 | drift=0.004 planar=n/a | body=0.974 cross=0.028 (n=100)
            diag: net=-0.005 ratio=n/a wz_ratio=n/a ripple=0.029 | drift=0.005 planar=n/a | body=1.016 cross=0.041 (n=100)
  turn_l_ceiling: net=0.162 ratio=0.962 wz_ratio=0.968 ripple=0.026 | drift=n/a planar=0.009 | body=n/a cross=n/a (n=100)
  turn_r_ceiling: net=-0.154 ratio=0.918 wz_ratio=0.921 ripple=0.046 | drift=n/a planar=0.010 | body=n/a cross=n/a (n=100)
     move_turn_l: net=0.052 ratio=0.622 wz_ratio=0.618 ripple=0.032 | drift=n/a planar=n/a | body=0.825 cross=0.027 (n=100)
     move_turn_r: net=-0.059 ratio=0.699 wz_ratio=0.699 ripple=0.063 | drift=n/a planar=n/a | body=0.878 cross=0.027 (n=100)

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.002 tripod=0.761 | closed=n/a progress=-0.018 net=-0.000 | drift=0.035 dir_err=n/a
        fwd: n=100 surv=1.000 (fwd=1.00) | ratio=0.993 achieved=0.248 tripod=0.680 | closed=n/a progress=-0.002 net=0.004 | drift=0.248 dir_err=0.016
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=1.043 achieved=0.229 tripod=0.714 | closed=n/a progress=0.001 net=0.001 | drift=0.232 dir_err=0.019
       back: n=100 surv=1.000 (back=1.00) | ratio=0.978 achieved=0.215 tripod=0.692 | closed=n/a progress=-0.001 net=-0.004 | drift=0.216 dir_err=0.033
       diag: n=100 surv=1.000 (diag=1.00) | ratio=1.011 achieved=0.253 tripod=0.693 | closed=n/a progress=0.003 net=0.003 | drift=0.257 dir_err=0.058
       turn: n=200 surv=1.000 (turn_l_ceiling=1.00 turn_r_ceiling=1.00) | ratio=n/a achieved=0.003 tripod=0.785 | closed=n/a progress=0.151 net=-0.019 | drift=0.043 dir_err=n/a
  move_turn: n=200 surv=1.000 (move_turn_l=1.00 move_turn_r=1.00) | ratio=0.888 achieved=0.195 tripod=0.701 | closed=n/a progress=0.058 net=-0.025 | drift=0.199 dir_err=0.025

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.144 action_fft_freq=1.639 joint_speed_mean=0.361 rod_speed_mean=0.029 contact_switch_hz=1.597 foot_slip=0.076 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.010 settle_six_down_s=12.000 foot_disp_end=0.119 work_per_s=298.893
        fwd: action_jitter=0.106 action_fft_freq=1.069 joint_speed_mean=0.754 rod_speed_mean=0.029 contact_switch_hz=1.840 foot_slip=0.093 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=22.292 slip_per_m=1.113 work_per_m=1115.388 cam_work_per_m=420.565
       side: action_jitter=0.083 action_fft_freq=1.129 joint_speed_mean=0.385 rod_speed_mean=0.029 contact_switch_hz=1.444 foot_slip=0.076 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=18.896 slip_per_m=1.020 work_per_m=1054.352 cam_work_per_m=194.404
       back: action_jitter=0.111 action_fft_freq=1.134 joint_speed_mean=0.704 rod_speed_mean=0.029 contact_switch_hz=1.806 foot_slip=0.093 wz_active_frac=0.073 clock_restarts=0.000 swings_per_m=25.393 slip_per_m=1.282 work_per_m=1222.408 cam_work_per_m=415.609
       diag: action_jitter=0.097 action_fft_freq=1.188 joint_speed_mean=0.574 rod_speed_mean=0.029 contact_switch_hz=1.674 foot_slip=0.088 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=20.076 slip_per_m=1.026 work_per_m=1023.167 cam_work_per_m=206.573
       turn: action_jitter=0.122 action_fft_freq=1.219 joint_speed_mean=0.680 rod_speed_mean=0.029 contact_switch_hz=1.667 foot_slip=0.076 wz_active_frac=0.923 clock_restarts=1.000 swings_per_rad=35.606 slip_per_rad=1.671 work_per_rad=1764.925 cam_work_per_rad=597.548
  move_turn: action_jitter=0.119 action_fft_freq=1.207 joint_speed_mean=0.667 rod_speed_mean=0.029 contact_switch_hz=1.861 foot_slip=0.088 wz_active_frac=0.838 clock_restarts=0.000 swings_per_rad=109.638 slip_per_rad=4.947 work_per_rad=5355.679 cam_work_per_rad=1588.422

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.206 steps_per_s=2.333 gait_hz=0.391 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=10.557 support_feet=2.961 taps_per_m=11.469 swing_apex=0.123 step_rate_hz=2.583 gait_hz=0.397 clock_hz=0.357 cadence_ratio=1.111 hz_speed_slope=-0.112 offset_R=0.061
       side: steps_per_m=11.375 support_feet=3.057 taps_per_m=7.429 swing_apex=0.104 step_rate_hz=2.583 gait_hz=0.406 clock_hz=0.314 cadence_ratio=1.291 hz_speed_slope=3.375 offset_R=0.061
       back: steps_per_m=12.876 support_feet=2.968 taps_per_m=12.074 swing_apex=0.111 step_rate_hz=2.750 gait_hz=0.409 clock_hz=0.320 cadence_ratio=1.265 hz_speed_slope=0.901 offset_R=0.109
       diag: steps_per_m=12.302 support_feet=2.938 taps_per_m=7.866 swing_apex=0.095 step_rate_hz=3.083 gait_hz=0.464 clock_hz=0.357 cadence_ratio=1.299 hz_speed_slope=0.465 offset_R=0.092
       turn: steps_per_rad=19.753 support_feet=3.134 taps_per_rad=15.343 swing_apex=0.130 step_rate_hz=2.750 gait_hz=0.464 clock_hz=0.455 cadence_ratio=1.022 steps_per_rad_pre=0.000 steps_per_rad_post=19.753 offset_R=0.098
  move_turn: steps_per_rad=51.651 support_feet=2.919 taps_per_rad=57.288 swing_apex=0.123 step_rate_hz=2.625 gait_hz=0.366 clock_hz=0.455 cadence_ratio=0.804 hz_speed_slope=-0.941 offset_R=0.060

body yaw by category (own hold, window [4 s, hold_s); medians pooled over both hands; per-hand values are in aggregate.suite.by_category[c].by_hand):
      stand: net_yaw_rate=0.002 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.015 heading_drift=0.003 planar_drift=n/a body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
        fwd: net_yaw_rate=-0.000 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.043 heading_drift=0.007 planar_drift=n/a body_ratio=0.988 cross_body=0.026 adapter_clipped_frac=n/a
       side: net_yaw_rate=-0.000 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.017 heading_drift=0.003 planar_drift=n/a body_ratio=1.046 cross_body=0.027 adapter_clipped_frac=n/a
       back: net_yaw_rate=0.002 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.040 heading_drift=0.004 planar_drift=n/a body_ratio=0.974 cross_body=0.028 adapter_clipped_frac=n/a
       diag: net_yaw_rate=-0.005 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.029 heading_drift=0.005 planar_drift=n/a body_ratio=1.016 cross_body=0.041 adapter_clipped_frac=n/a
       turn: net_yaw_rate=0.010 net_yaw_ratio=0.935 wz_ratio=0.937 yaw_ripple=0.034 heading_drift=n/a planar_drift=0.010 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  move_turn: net_yaw_rate=-0.001 net_yaw_ratio=0.658 wz_ratio=0.656 yaw_ripple=0.041 heading_drift=n/a planar_drift=n/a body_ratio=0.847 cross_body=0.027 adapter_clipped_frac=n/a

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
