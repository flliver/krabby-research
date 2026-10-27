```
=== crab-hex gait eval ===
scenario   : suite__omni_body  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-21_23-06-12_seed19_X06_p2a_A15pB/model_19994.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7495413993798157  p25=0.68039738382318  p75=0.7837478410651075
tippy_tap_fraction median=0.440668202764977
slip_ratio         median=0.22332930233104287
tracking_ratio     median=1.039177390376687  (achieved/commanded vx, walking holds; n=700)
schedule_completion_rate=1.0
terminations={'schedule_complete': 1000}
facing_error_rms   median=0.042680531964431126  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7726116535316999 (n=976)
     stand: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.7997880648898692 (n=100)
       fwd: cmd 0.25 -> achieved 0.262 m/s | tripod median=0.5886505494742287 (n=100)
     lat_l: cmd 0.22 -> achieved 0.234 m/s | tripod median=0.7277953159325574 (n=100)
     lat_r: cmd 0.22 -> achieved 0.235 m/s | tripod median=0.7950996888407479 (n=100)
      back: cmd 0.22 -> achieved 0.234 m/s | tripod median=0.7471552774663446 (n=100)
      diag: cmd 0.25 -> achieved 0.249 m/s | tripod median=0.7530284289094784 (n=100)
  turn_l_ceiling: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.7871986590014848 (n=100)
  turn_r_ceiling: cmd 0.00 -> achieved -0.001 m/s | tripod median=0.7922438108888857 (n=100)
  move_turn_l: cmd 0.22 -> achieved 0.195 m/s | tripod median=0.6202563189448318 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.214 m/s | tripod median=0.6700145480130941 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.017 initial=0.005 final=0.015 max=0.033 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.002 yaw_rate_ratio=n/a
     stand: err rms=0.026 initial=0.015 final=0.024 max=0.058 closed=n/a | settle=0.000 (n=100/100) | progress=-0.004 net=0.001 yaw_rate_ratio=0.437
       fwd: err rms=0.042 initial=0.017 final=0.041 max=0.088 closed=n/a | settle=0.000 (n=98/100) | progress=-0.002 net=-0.001 yaw_rate_ratio=0.294
     lat_l: err rms=0.026 initial=0.015 final=0.026 max=0.053 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a
     lat_r: err rms=0.023 initial=0.016 final=0.019 max=0.046 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
      back: err rms=0.033 initial=0.017 final=0.030 max=0.075 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.002 yaw_rate_ratio=n/a
      diag: err rms=0.038 initial=0.015 final=0.032 max=0.081 closed=n/a | settle=0.000 (n=100/100) | progress=-0.006 net=0.001 yaw_rate_ratio=0.351
  turn_l_ceiling: err rms=0.322 initial=0.015 final=0.422 max=0.464 closed=n/a | settle=n/a (n=0/100) | progress=0.142 net=0.032 yaw_rate_ratio=0.589
  turn_r_ceiling: err rms=0.367 initial=0.014 final=0.481 max=0.525 closed=n/a | settle=n/a (n=0/100) | progress=0.137 net=-0.037 yaw_rate_ratio=0.566
  move_turn_l: err rms=0.295 initial=0.010 final=0.461 max=0.467 closed=n/a | settle=n/a (n=0/100) | progress=0.051 net=0.037 yaw_rate_ratio=0.329
  move_turn_r: err rms=0.259 initial=0.015 final=0.377 max=0.395 closed=n/a | settle=n/a (n=0/100) | progress=0.060 net=-0.027 yaw_rate_ratio=0.298

body yaw by hold (window [4 s, hold_s); medians; net = net yaw rate rad/s, ratio = net / wz_cmd SIGNED (the gated turn number), wz_ratio = mean wz / wz_cmd; ripple = std of the 1-s mean wz; drift = |mean wz| on wz = 0 holds; planar = xy drift m/s on pure turns; body = along-bearing speed / |v_cmd|, cross = mean |perp| m/s):
          settle: net=n/a ratio=n/a wz_ratio=n/a ripple=n/a | drift=n/a planar=n/a | body=n/a cross=n/a (n=0)
           stand: net=0.000 ratio=n/a wz_ratio=n/a ripple=0.021 | drift=0.002 planar=n/a | body=n/a cross=n/a (n=100)
             fwd: net=0.004 ratio=n/a wz_ratio=n/a ripple=0.034 | drift=0.004 planar=n/a | body=1.044 cross=0.030 (n=100)
           lat_l: net=0.001 ratio=n/a wz_ratio=n/a ripple=0.012 | drift=0.002 planar=n/a | body=1.068 cross=0.022 (n=100)
           lat_r: net=0.001 ratio=n/a wz_ratio=n/a ripple=0.012 | drift=0.002 planar=n/a | body=1.065 cross=0.022 (n=100)
            back: net=-0.001 ratio=n/a wz_ratio=n/a ripple=0.021 | drift=0.005 planar=n/a | body=1.061 cross=0.023 (n=100)
            diag: net=0.004 ratio=n/a wz_ratio=n/a ripple=0.024 | drift=0.006 planar=n/a | body=0.998 cross=0.028 (n=100)
  turn_l_ceiling: net=0.148 ratio=0.883 wz_ratio=0.889 ripple=0.026 | drift=n/a planar=0.007 | body=n/a cross=n/a (n=100)
  turn_r_ceiling: net=-0.145 ratio=0.861 wz_ratio=0.869 ripple=0.024 | drift=n/a planar=0.009 | body=n/a cross=n/a (n=100)
     move_turn_l: net=0.053 ratio=0.635 wz_ratio=0.632 ripple=0.029 | drift=n/a planar=n/a | body=0.848 cross=0.029 (n=100)
     move_turn_r: net=-0.064 ratio=0.762 wz_ratio=0.759 ripple=0.033 | drift=n/a planar=n/a | body=0.946 cross=0.030 (n=100)

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.001 tripod=0.800 | closed=n/a progress=-0.004 net=0.001 | drift=0.038 dir_err=n/a
        fwd: n=100 surv=1.000 (fwd=1.00) | ratio=1.048 achieved=0.262 tripod=0.589 | closed=n/a progress=-0.002 net=-0.001 | drift=0.261 dir_err=0.065
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=1.066 achieved=0.234 tripod=0.763 | closed=n/a progress=n/a net=0.000 | drift=0.233 dir_err=0.014
       back: n=100 surv=1.000 (back=1.00) | ratio=1.064 achieved=0.234 tripod=0.747 | closed=n/a progress=n/a net=0.002 | drift=0.233 dir_err=0.046
       diag: n=100 surv=1.000 (diag=1.00) | ratio=0.996 achieved=0.249 tripod=0.753 | closed=n/a progress=-0.006 net=0.001 | drift=0.251 dir_err=0.032
       turn: n=200 surv=1.000 (turn_l_ceiling=1.00 turn_r_ceiling=1.00) | ratio=n/a achieved=-0.000 tripod=0.788 | closed=n/a progress=0.140 net=-0.033 | drift=0.046 dir_err=n/a
  move_turn: n=200 surv=1.000 (move_turn_l=1.00 move_turn_r=1.00) | ratio=0.926 achieved=0.204 tripod=0.650 | closed=n/a progress=0.056 net=0.025 | drift=0.207 dir_err=0.086

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.204 action_fft_freq=1.922 joint_speed_mean=0.336 rod_speed_mean=0.029 contact_switch_hz=1.604 foot_slip=0.068 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.003 settle_six_down_s=12.000 foot_disp_end=0.085 work_per_s=239.962
        fwd: action_jitter=0.153 action_fft_freq=1.233 joint_speed_mean=0.762 rod_speed_mean=0.029 contact_switch_hz=2.007 foot_slip=0.105 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=23.200 slip_per_m=1.186 work_per_m=1132.440 cam_work_per_m=405.679
       side: action_jitter=0.134 action_fft_freq=1.417 joint_speed_mean=0.344 rod_speed_mean=0.029 contact_switch_hz=1.583 foot_slip=0.073 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=20.406 slip_per_m=0.975 work_per_m=1029.766 cam_work_per_m=208.936
       back: action_jitter=0.166 action_fft_freq=1.282 joint_speed_mean=0.719 rod_speed_mean=0.029 contact_switch_hz=1.799 foot_slip=0.101 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=23.110 slip_per_m=1.281 work_per_m=1127.084 cam_work_per_m=409.986
       diag: action_jitter=0.157 action_fft_freq=1.341 joint_speed_mean=0.606 rod_speed_mean=0.029 contact_switch_hz=1.819 foot_slip=0.083 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=22.230 slip_per_m=1.021 work_per_m=994.450 cam_work_per_m=245.065
       turn: action_jitter=0.161 action_fft_freq=1.348 joint_speed_mean=0.657 rod_speed_mean=0.030 contact_switch_hz=1.583 foot_slip=0.086 wz_active_frac=0.925 clock_restarts=1.000 swings_per_rad=36.331 slip_per_rad=2.052 work_per_rad=1883.275 cam_work_per_rad=710.723
  move_turn: action_jitter=0.172 action_fft_freq=1.384 joint_speed_mean=0.666 rod_speed_mean=0.029 contact_switch_hz=1.993 foot_slip=0.099 wz_active_frac=0.853 clock_restarts=0.000 swings_per_rad=122.172 slip_per_rad=5.838 work_per_rad=6030.703 cam_work_per_rad=1793.073

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.055 steps_per_s=2.833 gait_hz=0.476 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=9.975 support_feet=2.942 taps_per_m=13.015 swing_apex=0.123 step_rate_hz=2.583 gait_hz=0.372 clock_hz=0.357 cadence_ratio=1.042 hz_speed_slope=-0.363 offset_R=0.079
       side: steps_per_m=12.675 support_feet=3.086 taps_per_m=7.496 swing_apex=0.086 step_rate_hz=2.917 gait_hz=0.488 clock_hz=0.314 cadence_ratio=1.554 hz_speed_slope=1.680 offset_R=0.028
       back: steps_per_m=10.909 support_feet=2.977 taps_per_m=11.952 swing_apex=0.124 step_rate_hz=2.542 gait_hz=0.403 clock_hz=0.314 cadence_ratio=1.282 hz_speed_slope=-0.923 offset_R=0.141
       diag: steps_per_m=12.448 support_feet=3.050 taps_per_m=9.548 swing_apex=0.097 step_rate_hz=3.083 gait_hz=0.500 clock_hz=0.357 cadence_ratio=1.401 hz_speed_slope=-0.142 offset_R=0.121
       turn: steps_per_rad=21.728 support_feet=3.098 taps_per_rad=14.582 swing_apex=0.115 step_rate_hz=2.833 gait_hz=0.482 clock_hz=0.457 cadence_ratio=1.055 steps_per_rad_pre=0.000 steps_per_rad_post=21.728 offset_R=0.017
  move_turn: steps_per_rad=50.850 support_feet=2.925 taps_per_rad=71.065 swing_apex=0.125 step_rate_hz=2.500 gait_hz=0.372 clock_hz=0.459 cadence_ratio=0.811 hz_speed_slope=0.925 offset_R=0.055

body yaw by category (own hold, window [4 s, hold_s); medians pooled over both hands; per-hand values are in aggregate.suite.by_category[c].by_hand):
      stand: net_yaw_rate=0.000 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.021 heading_drift=0.002 planar_drift=n/a body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
        fwd: net_yaw_rate=0.004 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.034 heading_drift=0.004 planar_drift=n/a body_ratio=1.044 cross_body=0.030 adapter_clipped_frac=n/a
       side: net_yaw_rate=0.001 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.012 heading_drift=0.002 planar_drift=n/a body_ratio=1.066 cross_body=0.022 adapter_clipped_frac=n/a
       back: net_yaw_rate=-0.001 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.021 heading_drift=0.005 planar_drift=n/a body_ratio=1.061 cross_body=0.023 adapter_clipped_frac=n/a
       diag: net_yaw_rate=0.004 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.024 heading_drift=0.006 planar_drift=n/a body_ratio=0.998 cross_body=0.028 adapter_clipped_frac=n/a
       turn: net_yaw_rate=0.004 net_yaw_ratio=0.878 wz_ratio=0.883 yaw_ripple=0.026 heading_drift=n/a planar_drift=0.007 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  move_turn: net_yaw_rate=-0.006 net_yaw_ratio=0.697 wz_ratio=0.698 yaw_ripple=0.031 heading_drift=n/a planar_drift=n/a body_ratio=0.888 cross_body=0.030 adapter_clipped_frac=n/a

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
