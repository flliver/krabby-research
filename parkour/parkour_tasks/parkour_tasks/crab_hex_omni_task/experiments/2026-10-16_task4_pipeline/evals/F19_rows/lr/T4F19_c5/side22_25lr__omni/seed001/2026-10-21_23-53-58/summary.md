```
=== crab-hex gait eval ===
scenario   : side22_25lr__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-10-21_19-09-27_seed19_X03_c5_A15pB/model_8997.pt
episodes   : 500  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7484379149320949  p25=0.7155238088759507  p75=0.7727520043771737
tippy_tap_fraction median=0.4598884598884599
slip_ratio         median=0.2594949402828194
tracking_ratio     median=0.9734752450054306  (achieved/commanded vx, walking holds; n=400)
schedule_completion_rate=1.0
terminations={'schedule_complete': 500}
facing_error_rms   median=0.014790943722979517  (rad, whole episode; n=500)

by hold:
    settle: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.41277997981734854 (n=429)
     stand: cmd 0.00 -> achieved 0.000 m/s | tripod median=0.39010822565543773 (n=100)
  lat_l_022: cmd 0.22 -> achieved 0.221 m/s | tripod median=0.8092568339918057 (n=100)
  lat_r_022: cmd 0.22 -> achieved 0.218 m/s | tripod median=0.7787754490222267 (n=100)
  lat_l_025: cmd 0.25 -> achieved 0.239 m/s | tripod median=0.7991392408344387 (n=100)
  lat_r_025: cmd 0.25 -> achieved 0.238 m/s | tripod median=0.7863598402631287 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.006 initial=0.004 final=0.006 max=0.011 closed=n/a | settle=0.000 (n=500/500) | progress=n/a net=0.000 yaw_rate_ratio=n/a
     stand: err rms=0.009 initial=0.006 final=0.012 max=0.020 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.000 yaw_rate_ratio=n/a
  lat_l_022: err rms=0.016 initial=0.006 final=0.017 max=0.036 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
  lat_r_022: err rms=0.018 initial=0.006 final=0.018 max=0.044 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a
  lat_l_025: err rms=0.016 initial=0.006 final=0.019 max=0.038 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
  lat_r_025: err rms=0.019 initial=0.006 final=0.025 max=0.044 closed=n/a | settle=0.000 (n=100/100) | progress=0.055 net=-0.000 yaw_rate_ratio=1.034

by category (5 waves, 500 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: n/a):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.000 tripod=0.390 | closed=n/a progress=n/a net=-0.000 | drift=0.031 dir_err=n/a
       side: n=400 surv=1.000 (lat_l_022=1.00 lat_r_022=1.00 lat_l_025=1.00 lat_r_025=1.00) | ratio=0.973 achieved=0.227 tripod=0.793 | closed=n/a progress=0.055 net=0.000 | drift=0.228 dir_err=0.018

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.263 action_fft_freq=2.956 joint_speed_mean=0.260 rod_speed_mean=0.029 contact_switch_hz=2.708 foot_slip=0.053 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.008 settle_six_down_s=12.000 foot_disp_end=0.087 work_per_s=227.272
       side: action_jitter=0.123 action_fft_freq=1.117 joint_speed_mean=0.336 rod_speed_mean=0.028 contact_switch_hz=1.306 foot_slip=0.113 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=17.151 slip_per_m=1.535 work_per_m=1361.341 cam_work_per_m=290.756

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.257 steps_per_s=3.500 gait_hz=1.114 clock_hz=0.000 cadence_ratio=n/a
       side: steps_per_m=9.197 support_feet=3.073 taps_per_m=7.773 swing_apex=0.099 step_rate_hz=2.083 gait_hz=0.336 clock_hz=0.336 cadence_ratio=0.991 hz_speed_slope=1.394 offset_R=0.994

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
