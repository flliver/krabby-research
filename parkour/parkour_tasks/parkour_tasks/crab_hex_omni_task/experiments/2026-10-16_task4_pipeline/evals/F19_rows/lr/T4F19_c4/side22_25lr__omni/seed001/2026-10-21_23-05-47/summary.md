```
=== crab-hex gait eval ===
scenario   : side22_25lr__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-10-21_18-21-11_seed19_X02_c4_A15pB/model_7498.pt
episodes   : 500  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7023690376749459  p25=0.6634638131503506  p75=0.7354201403696397
tippy_tap_fraction median=0.4384763278058159
slip_ratio         median=0.2580086121846917
tracking_ratio     median=0.9537846093089108  (achieved/commanded vx, walking holds; n=400)
schedule_completion_rate=1.0
terminations={'schedule_complete': 500}
facing_error_rms   median=0.015600535068985788  (rad, whole episode; n=500)

by hold:
    settle: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.4544784452850153 (n=438)
     stand: cmd 0.00 -> achieved -0.000 m/s | tripod median=0.45151209969734296 (n=100)
  lat_l_022: cmd 0.22 -> achieved 0.211 m/s | tripod median=0.7759760568555223 (n=100)
  lat_r_022: cmd 0.22 -> achieved 0.214 m/s | tripod median=0.7105621389289205 (n=100)
  lat_l_025: cmd 0.25 -> achieved 0.234 m/s | tripod median=0.7680271402693468 (n=100)
  lat_r_025: cmd 0.25 -> achieved 0.238 m/s | tripod median=0.7205309673172298 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.007 initial=0.007 final=0.006 max=0.011 closed=n/a | settle=0.000 (n=500/500) | progress=n/a net=0.001 yaw_rate_ratio=n/a
     stand: err rms=0.009 initial=0.006 final=0.008 max=0.019 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a
  lat_l_022: err rms=0.020 initial=0.005 final=0.027 max=0.046 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.001 yaw_rate_ratio=n/a
  lat_r_022: err rms=0.017 initial=0.007 final=0.021 max=0.044 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
  lat_l_025: err rms=0.020 initial=0.006 final=0.033 max=0.046 closed=n/a | settle=0.000 (n=99/100) | progress=-0.061 net=-0.001 yaw_rate_ratio=0.488
  lat_r_025: err rms=0.019 initial=0.006 final=0.025 max=0.049 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.002 yaw_rate_ratio=n/a

by category (5 waves, 500 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: n/a):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=-0.000 tripod=0.452 | closed=n/a progress=n/a net=0.000 | drift=0.032 dir_err=n/a
       side: n=400 surv=1.000 (lat_l_022=1.00 lat_r_022=1.00 lat_l_025=1.00 lat_r_025=1.00) | ratio=0.954 achieved=0.222 tripod=0.740 | closed=n/a progress=-0.061 net=-0.000 | drift=0.225 dir_err=0.019

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.227 action_fft_freq=2.632 joint_speed_mean=0.271 rod_speed_mean=0.029 contact_switch_hz=2.514 foot_slip=0.051 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.020 settle_six_down_s=12.000 foot_disp_end=0.072 work_per_s=231.915
       side: action_jitter=0.117 action_fft_freq=1.136 joint_speed_mean=0.350 rod_speed_mean=0.028 contact_switch_hz=1.389 foot_slip=0.105 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=18.654 slip_per_m=1.477 work_per_m=1309.401 cam_work_per_m=204.412

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.311 steps_per_s=3.667 gait_hz=1.099 clock_hz=0.000 cadence_ratio=n/a
       side: steps_per_m=9.737 support_feet=3.075 taps_per_m=8.671 swing_apex=0.110 step_rate_hz=2.167 gait_hz=0.336 clock_hz=0.336 cadence_ratio=0.991 hz_speed_slope=1.508 offset_R=0.991

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
