```
=== crab-hex gait eval ===
scenario   : side22_25lr__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-10-21_20-59-46_seed19_X05_c6x_A15pB/model_14995.pt
episodes   : 500  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7573059413683234  p25=0.7292319725351032  p75=0.7848566547667707
tippy_tap_fraction median=0.4066133720930233
slip_ratio         median=0.23406312188510275
tracking_ratio     median=1.016879680268164  (achieved/commanded vx, walking holds; n=400)
schedule_completion_rate=1.0
terminations={'schedule_complete': 500}
facing_error_rms   median=0.021504591500195985  (rad, whole episode; n=500)

by hold:
    settle: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.7529283415806444 (n=274)
     stand: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.7587390114644973 (n=100)
  lat_l_022: cmd 0.22 -> achieved 0.229 m/s | tripod median=0.7815499270557722 (n=100)
  lat_r_022: cmd 0.22 -> achieved 0.223 m/s | tripod median=0.7395371706451027 (n=100)
  lat_l_025: cmd 0.25 -> achieved 0.256 m/s | tripod median=0.7982005952484177 (n=100)
  lat_r_025: cmd 0.25 -> achieved 0.239 m/s | tripod median=0.707743463816909 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.014 initial=0.010 final=0.014 max=0.028 closed=n/a | settle=0.000 (n=500/500) | progress=n/a net=0.002 yaw_rate_ratio=n/a
     stand: err rms=0.020 initial=0.016 final=0.015 max=0.044 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
  lat_l_022: err rms=0.022 initial=0.013 final=0.019 max=0.048 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a
  lat_r_022: err rms=0.025 initial=0.016 final=0.019 max=0.053 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
  lat_l_025: err rms=0.024 initial=0.012 final=0.020 max=0.057 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
  lat_r_025: err rms=0.027 initial=0.014 final=0.023 max=0.059 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a

by category (5 waves, 500 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: n/a):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.001 tripod=0.759 | closed=n/a progress=n/a net=0.001 | drift=0.031 dir_err=n/a
       side: n=400 surv=1.000 (lat_l_022=1.00 lat_r_022=1.00 lat_l_025=1.00 lat_r_025=1.00) | ratio=1.017 achieved=0.231 tripod=0.757 | closed=n/a progress=n/a net=0.001 | drift=0.231 dir_err=0.013

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.246 action_fft_freq=2.356 joint_speed_mean=0.372 rod_speed_mean=0.026 contact_switch_hz=1.750 foot_slip=0.071 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.005 settle_six_down_s=12.000 foot_disp_end=0.175 work_per_s=249.240
       side: action_jitter=0.174 action_fft_freq=1.518 joint_speed_mean=0.343 rod_speed_mean=0.028 contact_switch_hz=1.653 foot_slip=0.094 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=20.991 slip_per_m=1.251 work_per_m=1064.007 cam_work_per_m=290.386

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.077 steps_per_s=3.250 gait_hz=0.543 clock_hz=0.000 cadence_ratio=n/a
       side: steps_per_m=11.604 support_feet=3.103 taps_per_m=9.002 swing_apex=0.080 step_rate_hz=2.833 gait_hz=0.375 clock_hz=0.336 cadence_ratio=1.091 hz_speed_slope=1.443 offset_R=0.028

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
