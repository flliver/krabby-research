```
=== crab-hex gait eval ===
scenario   : side22_25lr__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_28000.pt
episodes   : 500  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.71531785402167  p25=0.6693827474443937  p75=0.7511554432516008
tippy_tap_fraction median=0.39705882352941174
slip_ratio         median=0.20827169144476226
tracking_ratio     median=1.0247481815418584  (achieved/commanded vx, walking holds; n=400)
schedule_completion_rate=1.0
terminations={'schedule_complete': 500}
facing_error_rms   median=0.022977583671154267  (rad, whole episode; n=500)

by hold:
    settle: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.6673279587645602 (n=469)
     stand: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.7555256146540501 (n=100)
  lat_l_022: cmd 0.22 -> achieved 0.230 m/s | tripod median=0.6674047056410841 (n=100)
  lat_r_022: cmd 0.22 -> achieved 0.231 m/s | tripod median=0.7568624976764657 (n=100)
  lat_l_025: cmd 0.25 -> achieved 0.250 m/s | tripod median=0.6672560063490202 (n=100)
  lat_r_025: cmd 0.25 -> achieved 0.253 m/s | tripod median=0.7517589102473441 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.014 initial=0.008 final=0.016 max=0.030 closed=n/a | settle=0.000 (n=500/500) | progress=n/a net=0.002 yaw_rate_ratio=n/a
     stand: err rms=0.030 initial=0.014 final=0.032 max=0.062 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
  lat_l_022: err rms=0.019 initial=0.015 final=0.013 max=0.043 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.002 yaw_rate_ratio=n/a
  lat_r_022: err rms=0.035 initial=0.015 final=0.042 max=0.073 closed=n/a | settle=0.000 (n=100/100) | progress=0.000 net=0.000 yaw_rate_ratio=0.169
  lat_l_025: err rms=0.020 initial=0.018 final=0.018 max=0.046 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.002 yaw_rate_ratio=n/a
  lat_r_025: err rms=0.028 initial=0.016 final=0.027 max=0.063 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a

by category (5 waves, 500 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: n/a):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.002 tripod=0.756 | closed=n/a progress=n/a net=0.001 | drift=0.035 dir_err=n/a
       side: n=400 surv=1.000 (lat_l_022=1.00 lat_r_022=1.00 lat_l_025=1.00 lat_r_025=1.00) | ratio=1.025 achieved=0.239 tripod=0.713 | closed=n/a progress=0.000 net=0.001 | drift=0.241 dir_err=0.016

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.146 action_fft_freq=1.668 joint_speed_mean=0.359 rod_speed_mean=0.029 contact_switch_hz=1.583 foot_slip=0.075 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.012 settle_six_down_s=12.000 foot_disp_end=0.116 work_per_s=301.752
       side: action_jitter=0.080 action_fft_freq=1.107 joint_speed_mean=0.386 rod_speed_mean=0.029 contact_switch_hz=1.417 foot_slip=0.075 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=17.812 slip_per_m=0.957 work_per_m=997.993 cam_work_per_m=183.107

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.215 steps_per_s=2.333 gait_hz=0.391 clock_hz=0.000 cadence_ratio=n/a
       side: steps_per_m=10.884 support_feet=3.048 taps_per_m=6.963 swing_apex=0.097 step_rate_hz=2.667 gait_hz=0.409 clock_hz=0.336 cadence_ratio=1.204 hz_speed_slope=0.066 offset_R=0.048

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
