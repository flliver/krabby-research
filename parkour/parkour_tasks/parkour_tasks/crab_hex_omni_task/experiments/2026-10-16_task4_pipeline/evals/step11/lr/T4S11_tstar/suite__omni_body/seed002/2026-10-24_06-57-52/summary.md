```
=== crab-hex gait eval ===
scenario   : suite__omni_body  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7162770499220699  p25=0.6873848523311725  p75=0.770395644236025
tippy_tap_fraction median=0.4605263157894737
slip_ratio         median=0.24191206923384445
tracking_ratio     median=1.0044071465247468  (achieved/commanded vx, walking holds; n=700)
schedule_completion_rate=1.0
terminations={'schedule_complete': 1000}
facing_error_rms   median=0.04847237731802645  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.7711531276320129 (n=977)
     stand: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.787077889479137 (n=100)
       fwd: cmd 0.25 -> achieved 0.253 m/s | tripod median=0.6796974076824018 (n=100)
     lat_l: cmd 0.22 -> achieved 0.234 m/s | tripod median=0.6770280991287434 (n=100)
     lat_r: cmd 0.22 -> achieved 0.232 m/s | tripod median=0.7607958573300762 (n=100)
      back: cmd 0.22 -> achieved 0.217 m/s | tripod median=0.6944335095649917 (n=100)
      diag: cmd 0.25 -> achieved 0.254 m/s | tripod median=0.6914284604110372 (n=100)
  turn_l_ceiling: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.7864769535395397 (n=100)
  turn_r_ceiling: cmd 0.00 -> achieved 0.005 m/s | tripod median=0.7896882481909772 (n=100)
  move_turn_l: cmd 0.22 -> achieved 0.195 m/s | tripod median=0.6862778856049359 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.198 m/s | tripod median=0.7012758516017358 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.014 initial=0.008 final=0.012 max=0.028 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.001 yaw_rate_ratio=n/a
     stand: err rms=0.024 initial=0.013 final=0.028 max=0.055 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.044 initial=0.013 final=0.048 max=0.091 closed=n/a | settle=0.000 (n=100/100) | progress=0.000 net=0.001 yaw_rate_ratio=0.406
     lat_l: err rms=0.017 initial=0.010 final=0.014 max=0.040 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
     lat_r: err rms=0.034 initial=0.012 final=0.033 max=0.069 closed=n/a | settle=0.000 (n=100/100) | progress=-0.000 net=0.000 yaw_rate_ratio=0.187
      back: err rms=0.055 initial=0.012 final=0.072 max=0.122 closed=n/a | settle=0.000 (n=100/100) | progress=-0.001 net=0.001 yaw_rate_ratio=0.558
      diag: err rms=0.032 initial=0.013 final=0.027 max=0.074 closed=n/a | settle=0.000 (n=100/100) | progress=-0.008 net=0.001 yaw_rate_ratio=0.239
  turn_l_ceiling: err rms=0.211 initial=0.009 final=0.227 max=0.294 closed=n/a | settle=11.920 (n=3/100) | progress=0.160 net=0.001 yaw_rate_ratio=0.629
  turn_r_ceiling: err rms=0.279 initial=0.015 final=0.339 max=0.419 closed=n/a | settle=n/a (n=0/100) | progress=0.150 net=-0.023 yaw_rate_ratio=0.616
  move_turn_l: err rms=0.256 initial=0.011 final=0.405 max=0.417 closed=n/a | settle=n/a (n=0/100) | progress=0.055 net=-0.030 yaw_rate_ratio=0.290
  move_turn_r: err rms=0.274 initial=0.011 final=0.386 max=0.418 closed=n/a | settle=n/a (n=0/100) | progress=0.059 net=-0.026 yaw_rate_ratio=0.333

body yaw by hold (window [4 s, hold_s); medians; net = net yaw rate rad/s, ratio = net / wz_cmd SIGNED (the gated turn number), wz_ratio = mean wz / wz_cmd; ripple = std of the 1-s mean wz; drift = |mean wz| on wz = 0 holds; planar = xy drift m/s on pure turns; body = along-bearing speed / |v_cmd|, cross = mean |perp| m/s):
          settle: net=n/a ratio=n/a wz_ratio=n/a ripple=n/a | drift=n/a planar=n/a | body=n/a cross=n/a (n=0)
           stand: net=0.001 ratio=n/a wz_ratio=n/a ripple=0.015 | drift=0.003 planar=n/a | body=n/a cross=n/a (n=100)
             fwd: net=-0.002 ratio=n/a wz_ratio=n/a ripple=0.035 | drift=0.005 planar=n/a | body=1.007 cross=0.024 (n=100)
           lat_l: net=-0.000 ratio=n/a wz_ratio=n/a ripple=0.015 | drift=0.002 planar=n/a | body=1.064 cross=0.026 (n=100)
           lat_r: net=0.003 ratio=n/a wz_ratio=n/a ripple=0.019 | drift=0.004 planar=n/a | body=1.063 cross=0.027 (n=100)
            back: net=0.008 ratio=n/a wz_ratio=n/a ripple=0.034 | drift=0.008 planar=n/a | body=0.983 cross=0.027 (n=100)
            diag: net=-0.004 ratio=n/a wz_ratio=n/a ripple=0.030 | drift=0.004 planar=n/a | body=1.021 cross=0.038 (n=100)
  turn_l_ceiling: net=0.168 ratio=0.998 wz_ratio=1.003 ripple=0.023 | drift=n/a planar=0.005 | body=n/a cross=n/a (n=100)
  turn_r_ceiling: net=-0.152 ratio=0.908 wz_ratio=0.911 ripple=0.043 | drift=n/a planar=0.008 | body=n/a cross=n/a (n=100)
     move_turn_l: net=0.054 ratio=0.647 wz_ratio=0.638 ripple=0.028 | drift=n/a planar=n/a | body=0.843 cross=0.026 (n=100)
     move_turn_r: net=-0.059 ratio=0.699 wz_ratio=0.698 ripple=0.049 | drift=n/a planar=n/a | body=0.872 cross=0.027 (n=100)

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.004 tripod=0.787 | closed=n/a progress=n/a net=0.001 | drift=0.034 dir_err=n/a
        fwd: n=100 surv=1.000 (fwd=1.00) | ratio=1.010 achieved=0.253 tripod=0.680 | closed=n/a progress=0.000 net=0.001 | drift=0.252 dir_err=0.008
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=1.058 achieved=0.233 tripod=0.713 | closed=n/a progress=-0.000 net=0.001 | drift=0.234 dir_err=0.020
       back: n=100 surv=1.000 (back=1.00) | ratio=0.985 achieved=0.217 tripod=0.694 | closed=n/a progress=-0.001 net=0.001 | drift=0.218 dir_err=0.022
       diag: n=100 surv=1.000 (diag=1.00) | ratio=1.016 achieved=0.254 tripod=0.691 | closed=n/a progress=-0.008 net=0.001 | drift=0.257 dir_err=0.061
       turn: n=200 surv=1.000 (turn_l_ceiling=1.00 turn_r_ceiling=1.00) | ratio=n/a achieved=0.003 tripod=0.788 | closed=n/a progress=0.155 net=-0.015 | drift=0.042 dir_err=n/a
  move_turn: n=200 surv=1.000 (move_turn_l=1.00 move_turn_r=1.00) | ratio=0.896 achieved=0.197 tripod=0.692 | closed=n/a progress=0.057 net=-0.028 | drift=0.201 dir_err=0.014

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.204 action_fft_freq=1.972 joint_speed_mean=0.366 rod_speed_mean=0.029 contact_switch_hz=1.750 foot_slip=0.074 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.012 settle_six_down_s=12.000 foot_disp_end=0.091 work_per_s=290.758
        fwd: action_jitter=0.133 action_fft_freq=1.155 joint_speed_mean=0.752 rod_speed_mean=0.029 contact_switch_hz=1.889 foot_slip=0.094 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=22.691 slip_per_m=1.099 work_per_m=1087.715 cam_work_per_m=389.943
       side: action_jitter=0.108 action_fft_freq=1.241 joint_speed_mean=0.388 rod_speed_mean=0.029 contact_switch_hz=1.472 foot_slip=0.074 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=19.227 slip_per_m=0.982 work_per_m=1019.315 cam_work_per_m=194.054
       back: action_jitter=0.141 action_fft_freq=1.247 joint_speed_mean=0.706 rod_speed_mean=0.029 contact_switch_hz=1.917 foot_slip=0.094 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=27.020 slip_per_m=1.299 work_per_m=1208.028 cam_work_per_m=406.985
       diag: action_jitter=0.121 action_fft_freq=1.269 joint_speed_mean=0.578 rod_speed_mean=0.029 contact_switch_hz=1.750 foot_slip=0.089 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=20.815 slip_per_m=1.036 work_per_m=1002.464 cam_work_per_m=218.062
       turn: action_jitter=0.160 action_fft_freq=1.382 joint_speed_mean=0.672 rod_speed_mean=0.029 contact_switch_hz=1.750 foot_slip=0.074 wz_active_frac=0.923 clock_restarts=1.000 swings_per_rad=36.484 slip_per_rad=1.597 work_per_rad=1655.013 cam_work_per_rad=559.201
  move_turn: action_jitter=0.149 action_fft_freq=1.315 joint_speed_mean=0.670 rod_speed_mean=0.029 contact_switch_hz=1.979 foot_slip=0.088 wz_active_frac=0.844 clock_restarts=0.000 swings_per_rad=115.544 slip_per_rad=4.919 work_per_rad=5457.612 cam_work_per_rad=1627.552

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.171 steps_per_s=2.500 gait_hz=0.421 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=10.711 support_feet=2.942 taps_per_m=11.891 swing_apex=0.120 step_rate_hz=2.667 gait_hz=0.397 clock_hz=0.357 cadence_ratio=1.111 hz_speed_slope=0.114 offset_R=0.142
       side: steps_per_m=11.147 support_feet=3.063 taps_per_m=8.206 swing_apex=0.102 step_rate_hz=2.583 gait_hz=0.415 clock_hz=0.314 cadence_ratio=1.321 hz_speed_slope=3.155 offset_R=0.032
       back: steps_per_m=12.860 support_feet=2.967 taps_per_m=13.491 swing_apex=0.110 step_rate_hz=2.750 gait_hz=0.415 clock_hz=0.314 cadence_ratio=1.321 hz_speed_slope=0.289 offset_R=0.061
       diag: steps_per_m=11.987 support_feet=2.942 taps_per_m=8.623 swing_apex=0.093 step_rate_hz=3.000 gait_hz=0.470 clock_hz=0.357 cadence_ratio=1.316 hz_speed_slope=-0.214 offset_R=0.009
       turn: steps_per_rad=19.816 support_feet=3.117 taps_per_rad=16.286 swing_apex=0.126 step_rate_hz=2.833 gait_hz=0.476 clock_hz=0.449 cadence_ratio=1.061 steps_per_rad_pre=0.000 steps_per_rad_post=19.816 offset_R=0.015
  move_turn: steps_per_rad=51.927 support_feet=2.898 taps_per_rad=62.552 swing_apex=0.120 step_rate_hz=2.667 gait_hz=0.369 clock_hz=0.457 cadence_ratio=0.808 hz_speed_slope=0.308 offset_R=0.052

body yaw by category (own hold, window [4 s, hold_s); medians pooled over both hands; per-hand values are in aggregate.suite.by_category[c].by_hand):
      stand: net_yaw_rate=0.001 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.015 heading_drift=0.003 planar_drift=n/a body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
        fwd: net_yaw_rate=-0.002 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.035 heading_drift=0.005 planar_drift=n/a body_ratio=1.007 cross_body=0.024 adapter_clipped_frac=n/a
       side: net_yaw_rate=0.001 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.017 heading_drift=0.003 planar_drift=n/a body_ratio=1.063 cross_body=0.026 adapter_clipped_frac=n/a
       back: net_yaw_rate=0.008 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.034 heading_drift=0.008 planar_drift=n/a body_ratio=0.983 cross_body=0.027 adapter_clipped_frac=n/a
       diag: net_yaw_rate=-0.004 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.030 heading_drift=0.004 planar_drift=n/a body_ratio=1.021 cross_body=0.038 adapter_clipped_frac=n/a
       turn: net_yaw_rate=0.007 net_yaw_ratio=0.959 wz_ratio=0.964 yaw_ripple=0.031 heading_drift=n/a planar_drift=0.006 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  move_turn: net_yaw_rate=0.001 net_yaw_ratio=0.670 wz_ratio=0.664 yaw_ripple=0.033 heading_drift=n/a planar_drift=n/a body_ratio=0.857 cross_body=0.027 adapter_clipped_frac=n/a

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
