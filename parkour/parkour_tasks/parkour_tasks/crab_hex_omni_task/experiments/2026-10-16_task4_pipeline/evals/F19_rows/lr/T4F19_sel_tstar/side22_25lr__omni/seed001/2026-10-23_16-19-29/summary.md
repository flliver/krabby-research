```
=== crab-hex gait eval ===
scenario   : side22_25lr__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt
episodes   : 500  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7394971803542598  p25=0.697987944835634  p75=0.7656899034648477
tippy_tap_fraction median=0.43902439024390244
slip_ratio         median=0.22582544170200824
tracking_ratio     median=1.0320806327009948  (achieved/commanded vx, walking holds; n=400)
schedule_completion_rate=1.0
terminations={'schedule_complete': 500}
facing_error_rms   median=0.020755491853109223  (rad, whole episode; n=500)

by hold:
    settle: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7652573908969107 (n=487)
     stand: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7867098310677014 (n=100)
  lat_l_022: cmd 0.22 -> achieved 0.235 m/s | tripod median=0.6816408576968667 (n=100)
  lat_r_022: cmd 0.22 -> achieved 0.232 m/s | tripod median=0.7614959185091673 (n=100)
  lat_l_025: cmd 0.25 -> achieved 0.254 m/s | tripod median=0.6899032242624816 (n=100)
  lat_r_025: cmd 0.25 -> achieved 0.253 m/s | tripod median=0.7503891826013243 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.013 initial=0.008 final=0.012 max=0.028 closed=n/a | settle=0.000 (n=500/500) | progress=n/a net=0.002 yaw_rate_ratio=n/a
     stand: err rms=0.024 initial=0.010 final=0.023 max=0.055 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a
  lat_l_022: err rms=0.018 initial=0.011 final=0.014 max=0.042 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
  lat_r_022: err rms=0.031 initial=0.011 final=0.033 max=0.069 closed=n/a | settle=0.000 (n=100/100) | progress=0.005 net=0.001 yaw_rate_ratio=0.281
  lat_l_025: err rms=0.017 initial=0.013 final=0.014 max=0.041 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
  lat_r_025: err rms=0.026 initial=0.014 final=0.021 max=0.058 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a

by category (5 waves, 500 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: n/a):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.003 tripod=0.787 | closed=n/a progress=n/a net=0.000 | drift=0.034 dir_err=n/a
       side: n=400 surv=1.000 (lat_l_022=1.00 lat_r_022=1.00 lat_l_025=1.00 lat_r_025=1.00) | ratio=1.032 achieved=0.242 tripod=0.718 | closed=n/a progress=0.005 net=0.001 | drift=0.244 dir_err=0.017

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.208 action_fft_freq=1.997 joint_speed_mean=0.366 rod_speed_mean=0.029 contact_switch_hz=1.792 foot_slip=0.073 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.013 settle_six_down_s=12.000 foot_disp_end=0.084 work_per_s=287.699
       side: action_jitter=0.105 action_fft_freq=1.215 joint_speed_mean=0.390 rod_speed_mean=0.029 contact_switch_hz=1.472 foot_slip=0.074 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=18.515 slip_per_m=0.947 work_per_m=975.975 cam_work_per_m=183.536

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.152 steps_per_s=2.500 gait_hz=0.427 clock_hz=0.000 cadence_ratio=n/a
       side: steps_per_m=10.775 support_feet=3.065 taps_per_m=7.600 swing_apex=0.095 step_rate_hz=2.583 gait_hz=0.424 clock_hz=0.336 cadence_ratio=1.230 hz_speed_slope=0.171 offset_R=0.010

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
