```
=== crab-hex gait eval ===
scenario   : suite__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_03-27-53_seed19_X07_p3a_A15pB/model_22993.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6191623061659776  p25=0.5062741620207958  p75=0.6984647186873457
tippy_tap_fraction median=0.4362585771036475
slip_ratio         median=0.21000203924103966
tracking_ratio     median=0.9981138511537637  (achieved/commanded vx, walking holds; n=699)
schedule_completion_rate=0.926
terminations={'schedule_complete': 926, 'fall': 74}
facing_error_rms   median=0.0567711958542549  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.24112548971402087 (n=332)
     stand: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.5068678956956745 (n=100)
       fwd: cmd 0.25 -> achieved 0.257 m/s | tripod median=0.5941913418245003 (n=100)
     lat_l: cmd 0.22 -> achieved 0.217 m/s | tripod median=0.6324369000768508 (n=100)
     lat_r: cmd 0.22 -> achieved 0.235 m/s | tripod median=0.7476982512120738 (n=100)
      back: cmd 0.22 -> achieved 0.233 m/s | tripod median=0.613369010478049 (n=100)
      diag: cmd 0.25 -> achieved 0.246 m/s | tripod median=0.4667764694772483 (n=99)
  turn_l90: cmd 0.00 -> achieved -0.004 m/s | tripod median=0.7302983101464586 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7456432948480027 (n=100)
  move_turn: cmd 0.22 -> achieved 0.185 m/s | tripod median=0.39616192367100866 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.158 m/s | tripod median=0.6276782183720355 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.010 initial=0.006 final=0.010 max=0.021 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.002 yaw_rate_ratio=n/a
     stand: err rms=0.019 initial=0.011 final=0.019 max=0.046 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a
       fwd: err rms=0.054 initial=0.010 final=0.068 max=0.123 closed=n/a | settle=0.000 (n=95/100) | progress=-0.002 net=-0.000 yaw_rate_ratio=0.530
     lat_l: err rms=0.021 initial=0.009 final=0.018 max=0.050 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
     lat_r: err rms=0.014 initial=0.011 final=0.011 max=0.034 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
      back: err rms=0.059 initial=0.009 final=0.083 max=0.128 closed=n/a | settle=0.000 (n=94/100) | progress=-0.003 net=0.002 yaw_rate_ratio=0.387
      diag: err rms=0.046 initial=0.010 final=0.043 max=0.101 closed=n/a | settle=0.000 (n=93/100) | progress=-0.010 net=0.000 yaw_rate_ratio=0.471
  turn_l90: err rms=0.873 initial=1.570 final=0.147 max=1.574 closed=0.907 | settle=11.320 (n=53/100) | progress=0.123 net=0.119 yaw_rate_ratio=0.533
  turn_r90: err rms=0.875 initial=1.572 final=0.127 max=1.577 closed=0.919 | settle=11.320 (n=76/100) | progress=0.125 net=0.121 yaw_rate_ratio=0.533
  move_turn: err rms=1.246 initial=1.571 final=0.910 max=1.589 closed=0.419 | settle=n/a (n=0/100) | progress=0.058 net=0.058 yaw_rate_ratio=0.442
  move_turn_r: err rms=1.179 initial=1.572 final=0.810 max=1.575 closed=0.484 | settle=n/a (n=0/100) | progress=0.063 net=0.063 yaw_rate_ratio=0.449

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.001 tripod=0.507 | closed=n/a progress=n/a net=0.000 | drift=0.046 dir_err=n/a
        fwd: n=100 surv=0.980 (fwd=0.98) | ratio=1.028 achieved=0.257 tripod=0.594 | closed=n/a progress=-0.002 net=-0.000 | drift=0.258 dir_err=0.025
       side: n=200 surv=0.985 (lat_l=0.97 lat_r=1.00) | ratio=1.024 achieved=0.225 tripod=0.698 | closed=n/a progress=n/a net=0.001 | drift=0.228 dir_err=0.012
       back: n=100 surv=0.980 (back=0.98) | ratio=1.058 achieved=0.233 tripod=0.613 | closed=n/a progress=-0.003 net=0.002 | drift=0.233 dir_err=0.031
       diag: n=100 surv=0.660 (diag=0.66) | ratio=0.986 achieved=0.246 tripod=0.467 | closed=n/a progress=-0.010 net=0.000 | drift=0.253 dir_err=0.037
       turn: n=200 surv=1.000 (turn_l90=1.00 turn_r90=1.00) | ratio=n/a achieved=0.000 tripod=0.737 | closed=0.913 progress=0.125 net=0.119 | drift=0.048 dir_err=n/a
  move_turn: n=200 surv=0.835 (move_turn=0.67 move_turn_r=1.00) | ratio=0.766 achieved=0.168 tripod=0.565 | closed=0.460 progress=0.060 net=0.061 | drift=0.184 dir_err=0.056

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.167 action_fft_freq=1.684 joint_speed_mean=0.346 rod_speed_mean=0.029 contact_switch_hz=2.083 foot_slip=0.094 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.005 settle_six_down_s=12.000 foot_disp_end=0.220 work_per_s=330.779
        fwd: action_jitter=0.130 action_fft_freq=1.130 joint_speed_mean=0.727 rod_speed_mean=0.029 contact_switch_hz=1.806 foot_slip=0.116 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=21.172 slip_per_m=1.407 work_per_m=1430.206 cam_work_per_m=600.733
       side: action_jitter=0.096 action_fft_freq=1.081 joint_speed_mean=0.353 rod_speed_mean=0.028 contact_switch_hz=1.264 foot_slip=0.092 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=16.923 slip_per_m=1.335 work_per_m=1450.603 cam_work_per_m=221.584
       back: action_jitter=0.129 action_fft_freq=1.130 joint_speed_mean=0.703 rod_speed_mean=0.028 contact_switch_hz=1.792 foot_slip=0.115 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=23.449 slip_per_m=1.528 work_per_m=1481.951 cam_work_per_m=622.291
       diag: action_jitter=0.116 action_fft_freq=1.192 joint_speed_mean=0.616 rod_speed_mean=0.028 contact_switch_hz=2.049 foot_slip=0.177 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=25.754 slip_per_m=2.516 work_per_m=1621.761 cam_work_per_m=675.773
       turn: action_jitter=0.126 action_fft_freq=1.077 joint_speed_mean=0.689 rod_speed_mean=0.029 contact_switch_hz=1.417 foot_slip=0.096 wz_active_frac=0.960 clock_restarts=1.000 swings_per_rad=35.440 slip_per_rad=2.595 work_per_rad=2463.682 cam_work_per_rad=842.664
  move_turn: action_jitter=0.120 action_fft_freq=1.103 joint_speed_mean=0.645 rod_speed_mean=0.028 contact_switch_hz=1.611 foot_slip=0.127 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=80.348 slip_per_rad=6.409 work_per_rad=6336.558 cam_work_per_rad=2691.303

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.345 steps_per_s=2.917 gait_hz=0.409 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=10.717 support_feet=3.072 taps_per_m=10.364 swing_apex=0.101 step_rate_hz=2.750 gait_hz=0.385 clock_hz=0.357 cadence_ratio=1.077 hz_speed_slope=-1.595 offset_R=0.033
       side: steps_per_m=10.499 support_feet=3.246 taps_per_m=6.433 swing_apex=0.104 step_rate_hz=2.333 gait_hz=0.360 clock_hz=0.314 cadence_ratio=1.146 hz_speed_slope=0.526 offset_R=0.050
       back: steps_per_m=11.878 support_feet=3.071 taps_per_m=11.517 swing_apex=0.103 step_rate_hz=2.750 gait_hz=0.391 clock_hz=0.314 cadence_ratio=1.223 hz_speed_slope=-1.233 offset_R=0.054
       diag: steps_per_m=12.255 support_feet=3.410 taps_per_m=12.937 swing_apex=0.071 step_rate_hz=3.000 gait_hz=0.452 clock_hz=0.357 cadence_ratio=1.265 hz_speed_slope=-1.065 offset_R=0.090
       turn: steps_per_rad=21.463 support_feet=3.248 taps_per_rad=13.941 swing_apex=0.115 step_rate_hz=2.583 gait_hz=0.403 clock_hz=0.471 cadence_ratio=0.865 steps_per_rad_pre=19.991 steps_per_rad_post=2.013 offset_R=0.094
  move_turn: steps_per_rad=41.548 support_feet=3.106 taps_per_rad=35.677 swing_apex=0.110 step_rate_hz=2.583 gait_hz=0.360 clock_hz=0.500 cadence_ratio=0.720 hz_speed_slope=1.076 offset_R=0.040

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
