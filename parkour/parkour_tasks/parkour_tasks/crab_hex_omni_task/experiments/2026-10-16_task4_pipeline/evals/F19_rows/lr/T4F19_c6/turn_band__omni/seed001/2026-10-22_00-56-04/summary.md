```
=== crab-hex gait eval ===
scenario   : turn_band__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-10-21_19-57-38_seed19_X04_c6_A15pB/model_10996.pt
episodes   : 500  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.746488943817985  p25=0.7029869954833365  p75=0.7791515263695024
tippy_tap_fraction median=0.40103092783505156
slip_ratio         median=0.285097742683436
tracking_ratio     median=None  (achieved/commanded vx, walking holds; n=0)
schedule_completion_rate=1.0
terminations={'schedule_complete': 500}
facing_error_rms   median=0.20244963177433378  (rad, whole episode; n=500)

by hold:
    settle: cmd 0.00 -> achieved 0.000 m/s | tripod median=0.5246933032674387 (n=463)
     stand: cmd 0.00 -> achieved 0.000 m/s | tripod median=0.5104030314260849 (n=100)
  turn_l05: cmd 0.00 -> achieved -0.007 m/s | tripod median=0.790509342866509 (n=100)
  turn_r05: cmd 0.00 -> achieved 0.008 m/s | tripod median=0.7890493726511218 (n=100)
  turn_l10: cmd 0.00 -> achieved -0.025 m/s | tripod median=0.7666441172037228 (n=100)
  turn_r10: cmd 0.00 -> achieved 0.029 m/s | tripod median=0.797487958030763 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.007 initial=0.005 final=0.007 max=0.014 closed=n/a | settle=0.000 (n=500/500) | progress=n/a net=0.001 yaw_rate_ratio=n/a
     stand: err rms=0.012 initial=0.008 final=0.013 max=0.025 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.000 yaw_rate_ratio=n/a
  turn_l05: err rms=0.223 initial=0.496 final=0.120 max=0.497 closed=0.759 | settle=7.530 (n=100/100) | progress=0.043 net=0.031 yaw_rate_ratio=0.309
  turn_r05: err rms=0.225 initial=0.505 final=0.125 max=0.507 closed=0.753 | settle=7.720 (n=99/100) | progress=0.045 net=0.032 yaw_rate_ratio=0.329
  turn_l10: err rms=0.701 initial=0.995 final=0.503 max=0.995 closed=0.491 | settle=n/a (n=0/100) | progress=0.041 net=0.041 yaw_rate_ratio=0.291
  turn_r10: err rms=0.608 initial=1.003 final=0.262 max=1.003 closed=0.739 | settle=n/a (n=0/100) | progress=0.062 net=0.062 yaw_rate_ratio=0.305

by category (5 waves, 500 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: n/a):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.000 tripod=0.510 | closed=n/a progress=n/a net=-0.000 | drift=0.033 dir_err=n/a
       turn: n=200 surv=1.000 (turn_l05=1.00 turn_r05=1.00) | ratio=n/a achieved=-0.000 tripod=0.790 | closed=0.755 progress=0.044 net=0.032 | drift=0.037 dir_err=n/a
   turn_mid: n=200 surv=1.000 (turn_l10=1.00 turn_r10=1.00) | ratio=n/a achieved=0.008 tripod=0.786 | closed=0.647 progress=0.054 net=0.054 | drift=0.055 dir_err=n/a

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.286 action_fft_freq=3.089 joint_speed_mean=0.283 rod_speed_mean=0.029 contact_switch_hz=3.076 foot_slip=0.054 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.025 settle_six_down_s=12.000 foot_disp_end=0.102 work_per_s=230.446
       turn: action_jitter=0.235 action_fft_freq=2.138 joint_speed_mean=0.416 rod_speed_mean=0.028 contact_switch_hz=1.743 foot_slip=0.070 wz_active_frac=0.698 clock_restarts=4.000 swings_per_rad=165.299 slip_per_rad=6.989 work_per_rad=7395.363 cam_work_per_rad=1253.785
   turn_mid: 

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.376 steps_per_s=5.083 gait_hz=1.187 clock_hz=0.000 cadence_ratio=n/a
       turn: steps_per_rad=96.175 support_feet=3.164 taps_per_rad=65.896 swing_apex=0.095 step_rate_hz=3.083 gait_hz=0.464 clock_hz=0.318 cadence_ratio=1.436 steps_per_rad_pre=48.457 steps_per_rad_post=48.301 offset_R=0.995
   turn_mid: 

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
