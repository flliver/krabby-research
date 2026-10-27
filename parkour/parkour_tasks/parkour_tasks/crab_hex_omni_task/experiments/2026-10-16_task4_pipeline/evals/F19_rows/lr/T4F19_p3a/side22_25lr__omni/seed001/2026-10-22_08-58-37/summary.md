```
=== crab-hex gait eval ===
scenario   : side22_25lr__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_03-27-53_seed19_X07_p3a_A15pB/model_22993.pt
episodes   : 500  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7430427892091291  p25=0.7194786701630902  p75=0.7624198364915702
tippy_tap_fraction median=0.4122315592903828
slip_ratio         median=0.21940289471982616
tracking_ratio     median=1.0089291313708126  (achieved/commanded vx, walking holds; n=400)
schedule_completion_rate=1.0
terminations={'schedule_complete': 500}
facing_error_rms   median=0.017758995220007012  (rad, whole episode; n=500)

by hold:
    settle: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.15054158299499407 (n=170)
     stand: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.8064698650348501 (n=100)
  lat_l_022: cmd 0.22 -> achieved 0.210 m/s | tripod median=0.7353729330477339 (n=100)
  lat_r_022: cmd 0.22 -> achieved 0.243 m/s | tripod median=0.7497541746762506 (n=100)
  lat_l_025: cmd 0.25 -> achieved 0.227 m/s | tripod median=0.7369914068934352 (n=100)
  lat_r_025: cmd 0.25 -> achieved 0.265 m/s | tripod median=0.756367166422665 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.011 initial=0.008 final=0.013 max=0.023 closed=n/a | settle=0.000 (n=500/500) | progress=n/a net=0.001 yaw_rate_ratio=n/a
     stand: err rms=0.024 initial=0.010 final=0.018 max=0.061 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
  lat_l_022: err rms=0.019 initial=0.015 final=0.018 max=0.046 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.002 yaw_rate_ratio=n/a
  lat_r_022: err rms=0.015 initial=0.015 final=0.011 max=0.036 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.002 yaw_rate_ratio=n/a
  lat_l_025: err rms=0.022 initial=0.014 final=0.021 max=0.048 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
  lat_r_025: err rms=0.016 initial=0.013 final=0.015 max=0.036 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a

by category (5 waves, 500 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: n/a):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.003 tripod=0.806 | closed=n/a progress=n/a net=0.001 | drift=0.041 dir_err=n/a
       side: n=400 surv=1.000 (lat_l_022=1.00 lat_r_022=1.00 lat_l_025=1.00 lat_r_025=1.00) | ratio=1.009 achieved=0.236 tripod=0.745 | closed=n/a progress=n/a net=0.001 | drift=0.234 dir_err=0.010

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.187 action_fft_freq=1.779 joint_speed_mean=0.335 rod_speed_mean=0.029 contact_switch_hz=1.444 foot_slip=0.071 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.003 settle_six_down_s=12.000 foot_disp_end=0.215 work_per_s=249.281
       side: action_jitter=0.118 action_fft_freq=1.303 joint_speed_mean=0.364 rod_speed_mean=0.028 contact_switch_hz=1.444 foot_slip=0.072 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=18.915 slip_per_m=0.956 work_per_m=984.468 cam_work_per_m=143.854

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.063 steps_per_s=2.500 gait_hz=0.391 clock_hz=0.000 cadence_ratio=n/a
       side: steps_per_m=11.114 support_feet=3.073 taps_per_m=7.491 swing_apex=0.101 step_rate_hz=2.583 gait_hz=0.403 clock_hz=0.336 cadence_ratio=1.179 hz_speed_slope=-0.534 offset_R=0.045

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
