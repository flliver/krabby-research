```
=== crab-hex gait eval ===
scenario   : suite__omni_body  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7197812236650378  p25=0.685583289306685  p75=0.7719019017461064
tippy_tap_fraction median=0.4634146341463415
slip_ratio         median=0.2432486644187289
tracking_ratio     median=1.003175038039521  (achieved/commanded vx, walking holds; n=700)
schedule_completion_rate=1.0
terminations={'schedule_complete': 1000}
facing_error_rms   median=0.04994762452519618  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7657042576001434 (n=976)
     stand: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.790186608330925 (n=100)
       fwd: cmd 0.25 -> achieved 0.251 m/s | tripod median=0.6826047400175423 (n=100)
     lat_l: cmd 0.22 -> achieved 0.235 m/s | tripod median=0.6799541262174469 (n=100)
     lat_r: cmd 0.22 -> achieved 0.232 m/s | tripod median=0.7645436175008917 (n=100)
      back: cmd 0.22 -> achieved 0.217 m/s | tripod median=0.6959052031594617 (n=100)
      diag: cmd 0.25 -> achieved 0.254 m/s | tripod median=0.6884617450271318 (n=100)
  turn_l_ceiling: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.7848966551944083 (n=100)
  turn_r_ceiling: cmd 0.00 -> achieved 0.005 m/s | tripod median=0.7835321450124368 (n=100)
  move_turn_l: cmd 0.22 -> achieved 0.194 m/s | tripod median=0.6865634996511718 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.198 m/s | tripod median=0.6974957502147463 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.014 initial=0.008 final=0.013 max=0.028 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.002 yaw_rate_ratio=n/a
     stand: err rms=0.024 initial=0.013 final=0.027 max=0.055 closed=n/a | settle=0.000 (n=100/100) | progress=-0.001 net=0.000 yaw_rate_ratio=0.570
       fwd: err rms=0.042 initial=0.013 final=0.043 max=0.092 closed=n/a | settle=0.000 (n=99/100) | progress=-0.000 net=0.002 yaw_rate_ratio=0.355
     lat_l: err rms=0.017 initial=0.012 final=0.014 max=0.040 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
     lat_r: err rms=0.037 initial=0.013 final=0.040 max=0.080 closed=n/a | settle=0.000 (n=100/100) | progress=-0.000 net=0.000 yaw_rate_ratio=0.429
      back: err rms=0.055 initial=0.015 final=0.075 max=0.122 closed=n/a | settle=0.000 (n=100/100) | progress=-0.002 net=0.002 yaw_rate_ratio=0.572
      diag: err rms=0.031 initial=0.014 final=0.024 max=0.073 closed=n/a | settle=0.000 (n=100/100) | progress=0.003 net=0.001 yaw_rate_ratio=0.141
  turn_l_ceiling: err rms=0.212 initial=0.013 final=0.236 max=0.297 closed=n/a | settle=11.740 (n=2/100) | progress=0.159 net=-0.013 yaw_rate_ratio=0.635
  turn_r_ceiling: err rms=0.279 initial=0.012 final=0.344 max=0.416 closed=n/a | settle=n/a (n=0/100) | progress=0.150 net=-0.023 yaw_rate_ratio=0.620
  move_turn_l: err rms=0.260 initial=0.012 final=0.416 max=0.421 closed=n/a | settle=n/a (n=0/100) | progress=0.054 net=-0.028 yaw_rate_ratio=0.291
  move_turn_r: err rms=0.269 initial=0.013 final=0.379 max=0.409 closed=n/a | settle=n/a (n=0/100) | progress=0.060 net=-0.029 yaw_rate_ratio=0.333

body yaw by hold (window [4 s, hold_s); medians; net = net yaw rate rad/s, ratio = net / wz_cmd SIGNED (the gated turn number), wz_ratio = mean wz / wz_cmd; ripple = std of the 1-s mean wz; drift = |mean wz| on wz = 0 holds; planar = xy drift m/s on pure turns; body = along-bearing speed / |v_cmd|, cross = mean |perp| m/s):
          settle: net=n/a ratio=n/a wz_ratio=n/a ripple=n/a | drift=n/a planar=n/a | body=n/a cross=n/a (n=0)
           stand: net=0.001 ratio=n/a wz_ratio=n/a ripple=0.016 | drift=0.002 planar=n/a | body=n/a cross=n/a (n=100)
             fwd: net=-0.002 ratio=n/a wz_ratio=n/a ripple=0.031 | drift=0.005 planar=n/a | body=1.006 cross=0.025 (n=100)
           lat_l: net=-0.001 ratio=n/a wz_ratio=n/a ripple=0.014 | drift=0.002 planar=n/a | body=1.070 cross=0.025 (n=100)
           lat_r: net=0.004 ratio=n/a wz_ratio=n/a ripple=0.022 | drift=0.004 planar=n/a | body=1.063 cross=0.027 (n=100)
            back: net=0.008 ratio=n/a wz_ratio=n/a ripple=0.034 | drift=0.008 planar=n/a | body=0.982 cross=0.027 (n=100)
            diag: net=-0.003 ratio=n/a wz_ratio=n/a ripple=0.029 | drift=0.004 planar=n/a | body=1.019 cross=0.037 (n=100)
  turn_l_ceiling: net=0.164 ratio=0.977 wz_ratio=0.984 ripple=0.025 | drift=n/a planar=0.005 | body=n/a cross=n/a (n=100)
  turn_r_ceiling: net=-0.152 ratio=0.904 wz_ratio=0.908 ripple=0.044 | drift=n/a planar=0.008 | body=n/a cross=n/a (n=100)
     move_turn_l: net=0.054 ratio=0.639 wz_ratio=0.634 ripple=0.027 | drift=n/a planar=n/a | body=0.840 cross=0.026 (n=100)
     move_turn_r: net=-0.060 ratio=0.711 wz_ratio=0.706 ripple=0.051 | drift=n/a planar=n/a | body=0.875 cross=0.028 (n=100)

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.003 tripod=0.790 | closed=n/a progress=-0.001 net=0.000 | drift=0.034 dir_err=n/a
        fwd: n=100 surv=1.000 (fwd=1.00) | ratio=1.004 achieved=0.251 tripod=0.683 | closed=n/a progress=-0.000 net=0.002 | drift=0.251 dir_err=0.008
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=1.061 achieved=0.234 tripod=0.722 | closed=n/a progress=-0.000 net=0.001 | drift=0.234 dir_err=0.018
       back: n=100 surv=1.000 (back=1.00) | ratio=0.988 achieved=0.217 tripod=0.696 | closed=n/a progress=-0.002 net=0.002 | drift=0.218 dir_err=0.024
       diag: n=100 surv=1.000 (diag=1.00) | ratio=1.017 achieved=0.254 tripod=0.688 | closed=n/a progress=0.003 net=0.001 | drift=0.258 dir_err=0.064
       turn: n=200 surv=1.000 (turn_l_ceiling=1.00 turn_r_ceiling=1.00) | ratio=n/a achieved=0.004 tripod=0.784 | closed=n/a progress=0.155 net=-0.015 | drift=0.042 dir_err=n/a
  move_turn: n=200 surv=1.000 (move_turn_l=1.00 move_turn_r=1.00) | ratio=0.895 achieved=0.197 tripod=0.690 | closed=n/a progress=0.057 net=-0.029 | drift=0.201 dir_err=0.015

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.209 action_fft_freq=2.012 joint_speed_mean=0.364 rod_speed_mean=0.029 contact_switch_hz=1.722 foot_slip=0.072 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.012 settle_six_down_s=12.000 foot_disp_end=0.096 work_per_s=291.275
        fwd: action_jitter=0.133 action_fft_freq=1.163 joint_speed_mean=0.752 rod_speed_mean=0.029 contact_switch_hz=1.924 foot_slip=0.094 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=23.003 slip_per_m=1.108 work_per_m=1079.981 cam_work_per_m=385.396
       side: action_jitter=0.108 action_fft_freq=1.232 joint_speed_mean=0.390 rod_speed_mean=0.029 contact_switch_hz=1.514 foot_slip=0.074 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=19.603 slip_per_m=0.989 work_per_m=1016.738 cam_work_per_m=198.550
       back: action_jitter=0.141 action_fft_freq=1.247 joint_speed_mean=0.706 rod_speed_mean=0.029 contact_switch_hz=1.896 foot_slip=0.093 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=26.467 slip_per_m=1.268 work_per_m=1203.307 cam_work_per_m=403.283
       diag: action_jitter=0.120 action_fft_freq=1.267 joint_speed_mean=0.576 rod_speed_mean=0.029 contact_switch_hz=1.764 foot_slip=0.089 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=20.990 slip_per_m=1.027 work_per_m=1004.666 cam_work_per_m=210.411
       turn: action_jitter=0.159 action_fft_freq=1.378 joint_speed_mean=0.670 rod_speed_mean=0.029 contact_switch_hz=1.778 foot_slip=0.074 wz_active_frac=0.920 clock_restarts=1.000 swings_per_rad=37.120 slip_per_rad=1.610 work_per_rad=1655.002 cam_work_per_rad=562.158
  move_turn: action_jitter=0.149 action_fft_freq=1.321 joint_speed_mean=0.668 rod_speed_mean=0.029 contact_switch_hz=1.972 foot_slip=0.088 wz_active_frac=0.843 clock_restarts=0.000 swings_per_rad=115.331 slip_per_rad=5.048 work_per_rad=5451.945 cam_work_per_rad=1641.380

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.151 steps_per_s=2.500 gait_hz=0.427 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=10.770 support_feet=2.949 taps_per_m=12.236 swing_apex=0.120 step_rate_hz=2.667 gait_hz=0.397 clock_hz=0.357 cadence_ratio=1.111 hz_speed_slope=-0.127 offset_R=0.079
       side: steps_per_m=11.093 support_feet=3.063 taps_per_m=8.260 swing_apex=0.097 step_rate_hz=2.583 gait_hz=0.421 clock_hz=0.314 cadence_ratio=1.340 hz_speed_slope=3.953 offset_R=0.089
       back: steps_per_m=12.999 support_feet=2.962 taps_per_m=13.228 swing_apex=0.110 step_rate_hz=2.833 gait_hz=0.418 clock_hz=0.314 cadence_ratio=1.321 hz_speed_slope=0.336 offset_R=0.145
       diag: steps_per_m=11.862 support_feet=2.938 taps_per_m=8.915 swing_apex=0.094 step_rate_hz=3.000 gait_hz=0.470 clock_hz=0.357 cadence_ratio=1.316 hz_speed_slope=-0.303 offset_R=0.172
       turn: steps_per_rad=19.953 support_feet=3.112 taps_per_rad=16.912 swing_apex=0.125 step_rate_hz=2.833 gait_hz=0.476 clock_hz=0.447 cadence_ratio=1.065 steps_per_rad_pre=0.000 steps_per_rad_post=19.953 offset_R=0.076
  move_turn: steps_per_rad=51.938 support_feet=2.898 taps_per_rad=62.787 swing_apex=0.121 step_rate_hz=2.667 gait_hz=0.366 clock_hz=0.458 cadence_ratio=0.799 hz_speed_slope=-0.319 offset_R=0.039

body yaw by category (own hold, window [4 s, hold_s); medians pooled over both hands; per-hand values are in aggregate.suite.by_category[c].by_hand):
      stand: net_yaw_rate=0.001 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.016 heading_drift=0.002 planar_drift=n/a body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
        fwd: net_yaw_rate=-0.002 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.031 heading_drift=0.005 planar_drift=n/a body_ratio=1.006 cross_body=0.025 adapter_clipped_frac=n/a
       side: net_yaw_rate=0.001 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.018 heading_drift=0.003 planar_drift=n/a body_ratio=1.066 cross_body=0.026 adapter_clipped_frac=n/a
       back: net_yaw_rate=0.008 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.034 heading_drift=0.008 planar_drift=n/a body_ratio=0.982 cross_body=0.027 adapter_clipped_frac=n/a
       diag: net_yaw_rate=-0.003 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.029 heading_drift=0.004 planar_drift=n/a body_ratio=1.019 cross_body=0.037 adapter_clipped_frac=n/a
       turn: net_yaw_rate=0.006 net_yaw_ratio=0.947 wz_ratio=0.952 yaw_ripple=0.033 heading_drift=n/a planar_drift=0.006 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  move_turn: net_yaw_rate=-0.002 net_yaw_ratio=0.658 wz_ratio=0.654 yaw_ripple=0.034 heading_drift=n/a planar_drift=n/a body_ratio=0.861 cross_body=0.027 adapter_clipped_frac=n/a

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
