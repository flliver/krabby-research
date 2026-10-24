```
=== crab-hex gait eval ===
scenario   : suite__omni_body  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-21_23-06-12_seed19_X06_p2a_A15pB/model_19994.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7479636883791418  p25=0.6839055684897564  p75=0.7829819612084532
tippy_tap_fraction median=0.43559577677224737
slip_ratio         median=0.2243351063448787
tracking_ratio     median=1.037627263285897  (achieved/commanded vx, walking holds; n=700)
schedule_completion_rate=1.0
terminations={'schedule_complete': 1000}
facing_error_rms   median=0.045079487006748895  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.7797074167986702 (n=981)
     stand: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.7961421512757512 (n=100)
       fwd: cmd 0.25 -> achieved 0.262 m/s | tripod median=0.6045006537605616 (n=100)
     lat_l: cmd 0.22 -> achieved 0.234 m/s | tripod median=0.7268005759317675 (n=100)
     lat_r: cmd 0.22 -> achieved 0.234 m/s | tripod median=0.7945422639754535 (n=100)
      back: cmd 0.22 -> achieved 0.234 m/s | tripod median=0.7449866596585141 (n=100)
      diag: cmd 0.25 -> achieved 0.249 m/s | tripod median=0.754046924980134 (n=100)
  turn_l_ceiling: cmd 0.00 -> achieved -0.001 m/s | tripod median=0.7822227639738156 (n=100)
  turn_r_ceiling: cmd 0.00 -> achieved -0.002 m/s | tripod median=0.7921922625859569 (n=100)
  move_turn_l: cmd 0.22 -> achieved 0.196 m/s | tripod median=0.6155124087756894 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.213 m/s | tripod median=0.671800432347718 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.019 initial=0.006 final=0.015 max=0.034 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.001 yaw_rate_ratio=n/a
     stand: err rms=0.029 initial=0.013 final=0.028 max=0.062 closed=n/a | settle=0.000 (n=100/100) | progress=0.003 net=0.001 yaw_rate_ratio=0.279
       fwd: err rms=0.041 initial=0.016 final=0.043 max=0.088 closed=n/a | settle=0.000 (n=98/100) | progress=-0.007 net=-0.000 yaw_rate_ratio=0.250
     lat_l: err rms=0.025 initial=0.016 final=0.026 max=0.052 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a
     lat_r: err rms=0.021 initial=0.015 final=0.020 max=0.047 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a
      back: err rms=0.035 initial=0.015 final=0.035 max=0.077 closed=n/a | settle=0.000 (n=100/100) | progress=0.007 net=0.003 yaw_rate_ratio=0.282
      diag: err rms=0.037 initial=0.015 final=0.031 max=0.084 closed=n/a | settle=0.000 (n=100/100) | progress=-0.004 net=0.001 yaw_rate_ratio=0.183
  turn_l_ceiling: err rms=0.320 initial=0.017 final=0.430 max=0.468 closed=n/a | settle=n/a (n=0/100) | progress=0.142 net=0.032 yaw_rate_ratio=0.589
  turn_r_ceiling: err rms=0.354 initial=0.017 final=0.479 max=0.523 closed=n/a | settle=n/a (n=0/100) | progress=0.137 net=-0.036 yaw_rate_ratio=0.570
  move_turn_l: err rms=0.289 initial=0.013 final=0.465 max=0.469 closed=n/a | settle=n/a (n=0/100) | progress=0.050 net=0.036 yaw_rate_ratio=0.330
  move_turn_r: err rms=0.259 initial=0.016 final=0.381 max=0.393 closed=n/a | settle=n/a (n=0/100) | progress=0.059 net=-0.027 yaw_rate_ratio=0.298

body yaw by hold (window [4 s, hold_s); medians; net = net yaw rate rad/s, ratio = net / wz_cmd SIGNED (the gated turn number), wz_ratio = mean wz / wz_cmd; ripple = std of the 1-s mean wz; drift = |mean wz| on wz = 0 holds; planar = xy drift m/s on pure turns; body = along-bearing speed / |v_cmd|, cross = mean |perp| m/s):
          settle: net=n/a ratio=n/a wz_ratio=n/a ripple=n/a | drift=n/a planar=n/a | body=n/a cross=n/a (n=0)
           stand: net=0.001 ratio=n/a wz_ratio=n/a ripple=0.018 | drift=0.002 planar=n/a | body=n/a cross=n/a (n=100)
             fwd: net=0.003 ratio=n/a wz_ratio=n/a ripple=0.032 | drift=0.004 planar=n/a | body=1.046 cross=0.029 (n=100)
           lat_l: net=0.001 ratio=n/a wz_ratio=n/a ripple=0.012 | drift=0.002 planar=n/a | body=1.069 cross=0.023 (n=100)
           lat_r: net=0.000 ratio=n/a wz_ratio=n/a ripple=0.011 | drift=0.002 planar=n/a | body=1.067 cross=0.022 (n=100)
            back: net=-0.005 ratio=n/a wz_ratio=n/a ripple=0.022 | drift=0.007 planar=n/a | body=1.064 cross=0.023 (n=100)
            diag: net=0.004 ratio=n/a wz_ratio=n/a ripple=0.024 | drift=0.005 planar=n/a | body=1.001 cross=0.028 (n=100)
  turn_l_ceiling: net=0.148 ratio=0.879 wz_ratio=0.882 ripple=0.029 | drift=n/a planar=0.006 | body=n/a cross=n/a (n=100)
  turn_r_ceiling: net=-0.146 ratio=0.867 wz_ratio=0.870 ripple=0.026 | drift=n/a planar=0.009 | body=n/a cross=n/a (n=100)
     move_turn_l: net=0.053 ratio=0.629 wz_ratio=0.627 ripple=0.029 | drift=n/a planar=n/a | body=0.856 cross=0.030 (n=100)
     move_turn_r: net=-0.063 ratio=0.744 wz_ratio=0.742 ripple=0.032 | drift=n/a planar=n/a | body=0.944 cross=0.030 (n=100)

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.002 tripod=0.796 | closed=n/a progress=0.003 net=0.001 | drift=0.038 dir_err=n/a
        fwd: n=100 surv=1.000 (fwd=1.00) | ratio=1.048 achieved=0.262 tripod=0.605 | closed=n/a progress=-0.007 net=-0.000 | drift=0.260 dir_err=0.066
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=1.065 achieved=0.234 tripod=0.755 | closed=n/a progress=n/a net=0.000 | drift=0.233 dir_err=0.014
       back: n=100 surv=1.000 (back=1.00) | ratio=1.062 achieved=0.234 tripod=0.745 | closed=n/a progress=0.007 net=0.003 | drift=0.233 dir_err=0.047
       diag: n=100 surv=1.000 (diag=1.00) | ratio=0.997 achieved=0.249 tripod=0.754 | closed=n/a progress=-0.004 net=0.001 | drift=0.251 dir_err=0.034
       turn: n=200 surv=1.000 (turn_l_ceiling=1.00 turn_r_ceiling=1.00) | ratio=n/a achieved=-0.001 tripod=0.787 | closed=n/a progress=0.140 net=-0.031 | drift=0.046 dir_err=n/a
  move_turn: n=200 surv=1.000 (move_turn_l=1.00 move_turn_r=1.00) | ratio=0.939 achieved=0.206 tripod=0.648 | closed=n/a progress=0.055 net=-0.025 | drift=0.209 dir_err=0.089

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.207 action_fft_freq=1.931 joint_speed_mean=0.336 rod_speed_mean=0.029 contact_switch_hz=1.611 foot_slip=0.069 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.003 settle_six_down_s=12.000 foot_disp_end=0.086 work_per_s=239.291
        fwd: action_jitter=0.155 action_fft_freq=1.235 joint_speed_mean=0.761 rod_speed_mean=0.029 contact_switch_hz=2.014 foot_slip=0.104 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=23.187 slip_per_m=1.182 work_per_m=1116.235 cam_work_per_m=403.555
       side: action_jitter=0.135 action_fft_freq=1.416 joint_speed_mean=0.344 rod_speed_mean=0.029 contact_switch_hz=1.556 foot_slip=0.073 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=20.062 slip_per_m=0.971 work_per_m=1015.922 cam_work_per_m=203.143
       back: action_jitter=0.165 action_fft_freq=1.280 joint_speed_mean=0.719 rod_speed_mean=0.029 contact_switch_hz=1.792 foot_slip=0.100 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=23.050 slip_per_m=1.286 work_per_m=1129.683 cam_work_per_m=404.551
       diag: action_jitter=0.158 action_fft_freq=1.339 joint_speed_mean=0.607 rod_speed_mean=0.029 contact_switch_hz=1.806 foot_slip=0.084 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=21.778 slip_per_m=1.028 work_per_m=988.920 cam_work_per_m=242.935
       turn: action_jitter=0.161 action_fft_freq=1.351 joint_speed_mean=0.657 rod_speed_mean=0.030 contact_switch_hz=1.583 foot_slip=0.086 wz_active_frac=0.927 clock_restarts=1.000 swings_per_rad=36.851 slip_per_rad=2.072 work_per_rad=1895.073 cam_work_per_rad=713.168
  move_turn: action_jitter=0.172 action_fft_freq=1.381 joint_speed_mean=0.664 rod_speed_mean=0.029 contact_switch_hz=1.951 foot_slip=0.099 wz_active_frac=0.843 clock_restarts=0.000 swings_per_rad=119.015 slip_per_rad=5.872 work_per_rad=5880.302 cam_work_per_rad=1837.928

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.048 steps_per_s=2.833 gait_hz=0.476 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=10.013 support_feet=2.943 taps_per_m=13.179 swing_apex=0.125 step_rate_hz=2.583 gait_hz=0.372 clock_hz=0.357 cadence_ratio=1.042 hz_speed_slope=-0.250 offset_R=0.136
       side: steps_per_m=12.649 support_feet=3.085 taps_per_m=7.117 swing_apex=0.086 step_rate_hz=2.917 gait_hz=0.485 clock_hz=0.314 cadence_ratio=1.544 hz_speed_slope=0.938 offset_R=0.089
       back: steps_per_m=10.918 support_feet=2.977 taps_per_m=11.752 swing_apex=0.126 step_rate_hz=2.500 gait_hz=0.403 clock_hz=0.314 cadence_ratio=1.282 hz_speed_slope=-0.347 offset_R=0.191
       diag: steps_per_m=12.267 support_feet=3.051 taps_per_m=9.450 swing_apex=0.097 step_rate_hz=3.083 gait_hz=0.500 clock_hz=0.357 cadence_ratio=1.401 hz_speed_slope=0.338 offset_R=0.068
       turn: steps_per_rad=21.817 support_feet=3.098 taps_per_rad=14.790 swing_apex=0.116 step_rate_hz=2.833 gait_hz=0.482 clock_hz=0.457 cadence_ratio=1.055 steps_per_rad_pre=0.000 steps_per_rad_post=21.817 offset_R=0.038
  move_turn: steps_per_rad=51.268 support_feet=2.919 taps_per_rad=68.109 swing_apex=0.126 step_rate_hz=2.500 gait_hz=0.372 clock_hz=0.456 cadence_ratio=0.815 hz_speed_slope=0.840 offset_R=0.023

body yaw by category (own hold, window [4 s, hold_s); medians pooled over both hands; per-hand values are in aggregate.suite.by_category[c].by_hand):
      stand: net_yaw_rate=0.001 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.018 heading_drift=0.002 planar_drift=n/a body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
        fwd: net_yaw_rate=0.003 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.032 heading_drift=0.004 planar_drift=n/a body_ratio=1.046 cross_body=0.029 adapter_clipped_frac=n/a
       side: net_yaw_rate=0.001 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.012 heading_drift=0.002 planar_drift=n/a body_ratio=1.068 cross_body=0.022 adapter_clipped_frac=n/a
       back: net_yaw_rate=-0.005 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.022 heading_drift=0.007 planar_drift=n/a body_ratio=1.064 cross_body=0.023 adapter_clipped_frac=n/a
       diag: net_yaw_rate=0.004 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.024 heading_drift=0.005 planar_drift=n/a body_ratio=1.001 cross_body=0.028 adapter_clipped_frac=n/a
       turn: net_yaw_rate=0.001 net_yaw_ratio=0.870 wz_ratio=0.876 yaw_ripple=0.027 heading_drift=n/a planar_drift=0.007 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  move_turn: net_yaw_rate=-0.006 net_yaw_ratio=0.686 wz_ratio=0.683 yaw_ripple=0.030 heading_drift=n/a planar_drift=n/a body_ratio=0.905 cross_body=0.030 adapter_clipped_frac=n/a

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
