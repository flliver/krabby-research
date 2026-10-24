```
=== crab-hex gait eval ===
scenario   : turn_band__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-10-21_20-59-46_seed19_X05_c6x_A15pB/model_14995.pt
episodes   : 500  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7504667968644922  p25=0.7326159036870248  p75=0.7690488182514251
tippy_tap_fraction median=0.3684210526315789
slip_ratio         median=0.2914774484162539
tracking_ratio     median=None  (achieved/commanded vx, walking holds; n=0)
schedule_completion_rate=1.0
terminations={'schedule_complete': 500}
facing_error_rms   median=0.1573265705238073  (rad, whole episode; n=500)

by hold:
    settle: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.7502103855994653 (n=280)
     stand: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.7587390114644973 (n=100)
  turn_l05: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7475531600264455 (n=100)
  turn_r05: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.7547932953300303 (n=100)
  turn_l10: cmd 0.00 -> achieved 0.009 m/s | tripod median=0.7170489044907381 (n=100)
  turn_r10: cmd 0.00 -> achieved -0.020 m/s | tripod median=0.7675618019973365 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.015 initial=0.010 final=0.014 max=0.028 closed=n/a | settle=0.000 (n=500/500) | progress=n/a net=0.002 yaw_rate_ratio=n/a
     stand: err rms=0.020 initial=0.016 final=0.015 max=0.044 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
  turn_l05: err rms=0.175 initial=0.505 final=0.022 max=0.507 closed=0.957 | settle=3.030 (n=100/100) | progress=0.116 net=0.040 yaw_rate_ratio=0.451
  turn_r05: err rms=0.168 initial=0.495 final=0.017 max=0.495 closed=0.967 | settle=2.930 (n=100/100) | progress=0.118 net=0.041 yaw_rate_ratio=0.456
  turn_l10: err rms=0.576 initial=1.005 final=0.096 max=1.006 closed=0.905 | settle=10.420 (n=87/100) | progress=0.081 net=0.076 yaw_rate_ratio=0.418
  turn_r10: err rms=0.624 initial=0.994 final=0.212 max=0.994 closed=0.785 | settle=11.220 (n=36/100) | progress=0.066 net=0.065 yaw_rate_ratio=0.401

by category (5 waves, 500 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: n/a):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.001 tripod=0.759 | closed=n/a progress=n/a net=0.001 | drift=0.031 dir_err=n/a
       turn: n=200 surv=1.000 (turn_l05=1.00 turn_r05=1.00) | ratio=n/a achieved=0.002 tripod=0.751 | closed=0.962 progress=0.117 net=0.041 | drift=0.034 dir_err=n/a
   turn_mid: n=200 surv=1.000 (turn_l10=1.00 turn_r10=1.00) | ratio=n/a achieved=-0.006 tripod=0.745 | closed=0.877 progress=0.075 net=0.072 | drift=0.055 dir_err=n/a

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.246 action_fft_freq=2.356 joint_speed_mean=0.372 rod_speed_mean=0.026 contact_switch_hz=1.750 foot_slip=0.071 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.005 settle_six_down_s=12.000 foot_disp_end=0.175 work_per_s=249.240
       turn: action_jitter=0.233 action_fft_freq=2.159 joint_speed_mean=0.444 rod_speed_mean=0.026 contact_switch_hz=1.778 foot_slip=0.072 wz_active_frac=0.263 clock_restarts=0.000 swings_per_rad=132.401 slip_per_rad=5.487 work_per_rad=4808.361 cam_work_per_rad=1612.895
   turn_mid: 

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.077 steps_per_s=3.250 gait_hz=0.543 clock_hz=0.000 cadence_ratio=n/a
       turn: steps_per_rad=80.659 support_feet=3.084 taps_per_rad=49.981 swing_apex=0.089 step_rate_hz=3.250 gait_hz=0.537 clock_hz=0.126 cadence_ratio=4.282 steps_per_rad_pre=22.014 steps_per_rad_post=59.434 offset_R=0.150
   turn_mid: 

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
