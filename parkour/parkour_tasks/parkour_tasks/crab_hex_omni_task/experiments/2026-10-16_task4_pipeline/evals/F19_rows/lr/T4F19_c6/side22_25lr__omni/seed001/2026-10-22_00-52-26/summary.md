```
=== crab-hex gait eval ===
scenario   : side22_25lr__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-10-21_19-57-38_seed19_X04_c6_A15pB/model_10996.pt
episodes   : 500  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7535118904381948  p25=0.7204792988014237  p75=0.7769750166861777
tippy_tap_fraction median=0.44220647773279353
slip_ratio         median=0.24923841045865042
tracking_ratio     median=0.9899995966716131  (achieved/commanded vx, walking holds; n=400)
schedule_completion_rate=1.0
terminations={'schedule_complete': 500}
facing_error_rms   median=0.017314287494863537  (rad, whole episode; n=500)

by hold:
    settle: cmd 0.00 -> achieved 0.000 m/s | tripod median=0.5246933032674387 (n=463)
     stand: cmd 0.00 -> achieved 0.000 m/s | tripod median=0.5104030314260849 (n=100)
  lat_l_022: cmd 0.22 -> achieved 0.219 m/s | tripod median=0.8128617537035485 (n=100)
  lat_r_022: cmd 0.22 -> achieved 0.219 m/s | tripod median=0.7685610814847766 (n=100)
  lat_l_025: cmd 0.25 -> achieved 0.247 m/s | tripod median=0.7894265534030489 (n=100)
  lat_r_025: cmd 0.25 -> achieved 0.246 m/s | tripod median=0.7878707318148974 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.007 initial=0.005 final=0.008 max=0.014 closed=n/a | settle=0.000 (n=500/500) | progress=n/a net=0.001 yaw_rate_ratio=n/a
     stand: err rms=0.012 initial=0.008 final=0.013 max=0.025 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.000 yaw_rate_ratio=n/a
  lat_l_022: err rms=0.017 initial=0.007 final=0.015 max=0.047 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
  lat_r_022: err rms=0.026 initial=0.008 final=0.031 max=0.059 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.000 yaw_rate_ratio=n/a
  lat_l_025: err rms=0.017 initial=0.007 final=0.015 max=0.043 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
  lat_r_025: err rms=0.027 initial=0.008 final=0.032 max=0.060 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.000 yaw_rate_ratio=n/a

by category (5 waves, 500 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: n/a):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.000 tripod=0.510 | closed=n/a progress=n/a net=-0.000 | drift=0.033 dir_err=n/a
       side: n=400 surv=1.000 (lat_l_022=1.00 lat_r_022=1.00 lat_l_025=1.00 lat_r_025=1.00) | ratio=0.990 achieved=0.227 tripod=0.788 | closed=n/a progress=n/a net=0.000 | drift=0.228 dir_err=0.014

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.286 action_fft_freq=3.089 joint_speed_mean=0.283 rod_speed_mean=0.029 contact_switch_hz=3.076 foot_slip=0.054 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.025 settle_six_down_s=12.000 foot_disp_end=0.102 work_per_s=230.446
       side: action_jitter=0.146 action_fft_freq=1.222 joint_speed_mean=0.332 rod_speed_mean=0.028 contact_switch_hz=1.333 foot_slip=0.103 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=17.393 slip_per_m=1.415 work_per_m=1203.267 cam_work_per_m=294.456

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.376 steps_per_s=5.083 gait_hz=1.187 clock_hz=0.000 cadence_ratio=n/a
       side: steps_per_m=9.459 support_feet=3.100 taps_per_m=7.801 swing_apex=0.100 step_rate_hz=2.167 gait_hz=0.333 clock_hz=0.336 cadence_ratio=0.991 hz_speed_slope=1.172 offset_R=0.994

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
