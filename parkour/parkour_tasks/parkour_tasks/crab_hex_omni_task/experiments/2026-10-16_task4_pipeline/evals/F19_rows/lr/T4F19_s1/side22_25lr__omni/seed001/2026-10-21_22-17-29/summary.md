```
=== crab-hex gait eval ===
scenario   : side22_25lr__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-10-21_15-52-28_seed19_X01_s1_A15pB/model_5999.pt
episodes   : 500  unscored: 1
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7169245023133275  p25=0.6624217134237484  p75=0.7448117736813942
tippy_tap_fraction median=0.5076995972518361
slip_ratio         median=0.28520913811796833
tracking_ratio     median=0.8464362147502649  (achieved/commanded vx, walking holds; n=400)
schedule_completion_rate=1.0
terminations={'schedule_complete': 500}
facing_error_rms   median=0.019560537310146807  (rad, whole episode; n=500)

by hold:
    settle: cmd 0.00 -> achieved -0.003 m/s | tripod median=0.064710100836817 (n=445)
     stand: cmd 0.00 -> achieved -0.001 m/s | tripod median=0.19442710576186956 (n=98)
  lat_l_022: cmd 0.22 -> achieved 0.191 m/s | tripod median=0.7877616709559606 (n=100)
  lat_r_022: cmd 0.22 -> achieved 0.186 m/s | tripod median=0.7556224179704094 (n=100)
  lat_l_025: cmd 0.25 -> achieved 0.208 m/s | tripod median=0.7993363126541669 (n=100)
  lat_r_025: cmd 0.25 -> achieved 0.209 m/s | tripod median=0.7319922794238097 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.007 initial=0.006 final=0.006 max=0.012 closed=n/a | settle=0.000 (n=500/500) | progress=n/a net=0.001 yaw_rate_ratio=n/a
     stand: err rms=0.011 initial=0.005 final=0.014 max=0.022 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.000 yaw_rate_ratio=n/a
  lat_l_022: err rms=0.021 initial=0.006 final=0.025 max=0.051 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.000 yaw_rate_ratio=n/a
  lat_r_022: err rms=0.027 initial=0.006 final=0.044 max=0.056 closed=n/a | settle=0.000 (n=99/100) | progress=-0.003 net=0.001 yaw_rate_ratio=0.263
  lat_l_025: err rms=0.019 initial=0.007 final=0.025 max=0.044 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.000 yaw_rate_ratio=n/a
  lat_r_025: err rms=0.034 initial=0.007 final=0.051 max=0.067 closed=n/a | settle=0.000 (n=100/100) | progress=-0.011 net=-0.001 yaw_rate_ratio=0.284

by category (5 waves, 500 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: n/a):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=-0.001 tripod=0.194 | closed=n/a progress=n/a net=-0.000 | drift=0.031 dir_err=n/a
       side: n=400 surv=1.000 (lat_l_022=1.00 lat_r_022=1.00 lat_l_025=1.00 lat_r_025=1.00) | ratio=0.846 achieved=0.197 tripod=0.774 | closed=n/a progress=-0.004 net=-0.000 | drift=0.200 dir_err=0.018

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.296 action_fft_freq=3.752 joint_speed_mean=0.254 rod_speed_mean=0.029 contact_switch_hz=2.799 foot_slip=0.052 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.008 settle_six_down_s=12.000 foot_disp_end=0.058 work_per_s=224.415
       side: action_jitter=0.126 action_fft_freq=1.172 joint_speed_mean=0.317 rod_speed_mean=0.028 contact_switch_hz=1.347 foot_slip=0.091 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=20.528 slip_per_m=1.452 work_per_m=1344.416 cam_work_per_m=313.383

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.405 steps_per_s=3.167 gait_hz=1.205 clock_hz=0.000 cadence_ratio=n/a
       side: steps_per_m=10.780 support_feet=3.067 taps_per_m=9.523 swing_apex=0.103 step_rate_hz=2.167 gait_hz=0.330 clock_hz=0.341 cadence_ratio=0.991 hz_speed_slope=1.534 offset_R=0.995

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
