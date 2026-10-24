```
=== crab-hex gait eval ===
scenario   : side22_25lr__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-21_23-06-12_seed19_X06_p2a_A15pB/model_19994.pt
episodes   : 500  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7637380467200039  p25=0.7258640438466945  p75=0.7889341940375458
tippy_tap_fraction median=0.4
slip_ratio         median=0.1929995696358285
tracking_ratio     median=1.0435011819175284  (achieved/commanded vx, walking holds; n=400)
schedule_completion_rate=1.0
terminations={'schedule_complete': 500}
facing_error_rms   median=0.02248983460742878  (rad, whole episode; n=500)

by hold:
    settle: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7736777446353714 (n=485)
     stand: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.797374791651819 (n=100)
  lat_l_022: cmd 0.22 -> achieved 0.234 m/s | tripod median=0.7221482950991249 (n=100)
  lat_r_022: cmd 0.22 -> achieved 0.234 m/s | tripod median=0.7926664323077026 (n=100)
  lat_l_025: cmd 0.25 -> achieved 0.252 m/s | tripod median=0.708959055390698 (n=100)
  lat_r_025: cmd 0.25 -> achieved 0.257 m/s | tripod median=0.7801690182948503 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.016 initial=0.005 final=0.013 max=0.033 closed=n/a | settle=0.000 (n=500/500) | progress=n/a net=0.002 yaw_rate_ratio=n/a
     stand: err rms=0.024 initial=0.014 final=0.021 max=0.055 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a
  lat_l_022: err rms=0.023 initial=0.014 final=0.025 max=0.051 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
  lat_r_022: err rms=0.023 initial=0.012 final=0.026 max=0.048 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a
  lat_l_025: err rms=0.026 initial=0.014 final=0.023 max=0.055 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
  lat_r_025: err rms=0.025 initial=0.014 final=0.027 max=0.050 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a

by category (5 waves, 500 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: n/a):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.002 tripod=0.797 | closed=n/a progress=n/a net=0.000 | drift=0.038 dir_err=n/a
       side: n=400 surv=1.000 (lat_l_022=1.00 lat_r_022=1.00 lat_l_025=1.00 lat_r_025=1.00) | ratio=1.044 achieved=0.241 tripod=0.748 | closed=n/a progress=n/a net=0.000 | drift=0.240 dir_err=0.015

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.203 action_fft_freq=1.898 joint_speed_mean=0.337 rod_speed_mean=0.030 contact_switch_hz=1.583 foot_slip=0.068 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.003 settle_six_down_s=12.000 foot_disp_end=0.081 work_per_s=238.767
       side: action_jitter=0.129 action_fft_freq=1.379 joint_speed_mean=0.344 rod_speed_mean=0.029 contact_switch_hz=1.597 foot_slip=0.075 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=19.934 slip_per_m=0.970 work_per_m=980.115 cam_work_per_m=198.635

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.057 steps_per_s=2.833 gait_hz=0.482 clock_hz=0.000 cadence_ratio=n/a
       side: steps_per_m=12.228 support_feet=3.090 taps_per_m=7.317 swing_apex=0.081 step_rate_hz=3.000 gait_hz=0.494 clock_hz=0.336 cadence_ratio=1.495 hz_speed_slope=-0.191 offset_R=0.023

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
