```
=== crab-hex gait eval ===
scenario   : side22_25lr__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt
episodes   : 500  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7413694902042112  p25=0.695957182867066  p75=0.7684884820927107
tippy_tap_fraction median=0.43548387096774194
slip_ratio         median=0.2240291217491539
tracking_ratio     median=1.035087047458606  (achieved/commanded vx, walking holds; n=400)
schedule_completion_rate=1.0
terminations={'schedule_complete': 500}
facing_error_rms   median=0.020666156507932625  (rad, whole episode; n=500)

by hold:
    settle: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.7706569413525856 (n=487)
     stand: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7908561719938312 (n=100)
  lat_l_022: cmd 0.22 -> achieved 0.235 m/s | tripod median=0.6809711693007171 (n=100)
  lat_r_022: cmd 0.22 -> achieved 0.232 m/s | tripod median=0.7633064409032326 (n=100)
  lat_l_025: cmd 0.25 -> achieved 0.254 m/s | tripod median=0.6879514909560198 (n=100)
  lat_r_025: cmd 0.25 -> achieved 0.253 m/s | tripod median=0.7513651722216091 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.014 initial=0.008 final=0.013 max=0.028 closed=n/a | settle=0.000 (n=500/500) | progress=n/a net=0.002 yaw_rate_ratio=n/a
     stand: err rms=0.023 initial=0.012 final=0.020 max=0.054 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
  lat_l_022: err rms=0.018 initial=0.013 final=0.013 max=0.043 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
  lat_r_022: err rms=0.030 initial=0.012 final=0.030 max=0.068 closed=n/a | settle=0.000 (n=100/100) | progress=-0.005 net=0.001 yaw_rate_ratio=0.582
  lat_l_025: err rms=0.017 initial=0.014 final=0.011 max=0.042 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
  lat_r_025: err rms=0.028 initial=0.013 final=0.024 max=0.063 closed=n/a | settle=0.000 (n=100/100) | progress=0.004 net=0.001 yaw_rate_ratio=0.254

by category (5 waves, 500 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: n/a):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.003 tripod=0.791 | closed=n/a progress=n/a net=0.001 | drift=0.034 dir_err=n/a
       side: n=400 surv=1.000 (lat_l_022=1.00 lat_r_022=1.00 lat_l_025=1.00 lat_r_025=1.00) | ratio=1.035 achieved=0.242 tripod=0.723 | closed=n/a progress=-0.000 net=0.001 | drift=0.244 dir_err=0.019

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.207 action_fft_freq=1.996 joint_speed_mean=0.366 rod_speed_mean=0.029 contact_switch_hz=1.750 foot_slip=0.073 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.012 settle_six_down_s=12.000 foot_disp_end=0.091 work_per_s=291.953
       side: action_jitter=0.104 action_fft_freq=1.206 joint_speed_mean=0.391 rod_speed_mean=0.029 contact_switch_hz=1.500 foot_slip=0.074 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=18.353 slip_per_m=0.943 work_per_m=975.078 cam_work_per_m=185.214

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.162 steps_per_s=2.500 gait_hz=0.421 clock_hz=0.000 cadence_ratio=n/a
       side: steps_per_m=10.832 support_feet=3.062 taps_per_m=7.687 swing_apex=0.094 step_rate_hz=2.583 gait_hz=0.427 clock_hz=0.336 cadence_ratio=1.243 hz_speed_slope=0.201 offset_R=0.051

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
