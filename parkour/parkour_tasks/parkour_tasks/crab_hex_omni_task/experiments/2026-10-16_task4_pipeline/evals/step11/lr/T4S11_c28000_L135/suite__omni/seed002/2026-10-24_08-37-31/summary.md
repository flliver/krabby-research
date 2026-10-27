```
=== crab-hex gait eval ===
scenario   : suite__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_28000.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6642796790557364  p25=0.612934269625375  p75=0.712502658550829
tippy_tap_fraction median=0.359375
slip_ratio         median=0.19511451082425507
tracking_ratio     median=1.0183285892184264  (achieved/commanded vx, walking holds; n=700)
schedule_completion_rate=0.998
terminations={'schedule_complete': 998, 'fall': 2}
facing_error_rms   median=0.057881529205418135  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.4982505707444589 (n=902)
     stand: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7296656304128 (n=100)
       fwd: cmd 0.25 -> achieved 0.254 m/s | tripod median=0.6400433810922201 (n=100)
     lat_l: cmd 0.22 -> achieved 0.230 m/s | tripod median=0.6520191931399275 (n=100)
     lat_r: cmd 0.22 -> achieved 0.229 m/s | tripod median=0.7407071222883157 (n=100)
      back: cmd 0.22 -> achieved 0.224 m/s | tripod median=0.6820300487557626 (n=100)
      diag: cmd 0.25 -> achieved 0.259 m/s | tripod median=0.6662994868615484 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.010 m/s | tripod median=0.7603962656184087 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.008 m/s | tripod median=0.760354724269922 (n=100)
  move_turn: cmd 0.22 -> achieved 0.178 m/s | tripod median=0.5393352863460045 (n=100)
  move_turn_r: cmd 0.22 -> achieved 0.172 m/s | tripod median=0.627333699984162 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.012 initial=0.010 final=0.015 max=0.026 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.001 yaw_rate_ratio=n/a
     stand: err rms=0.030 initial=0.016 final=0.034 max=0.063 closed=n/a | settle=0.000 (n=100/100) | progress=0.006 net=-0.000 yaw_rate_ratio=0.298
       fwd: err rms=0.042 initial=0.016 final=0.055 max=0.107 closed=n/a | settle=0.000 (n=100/100) | progress=-0.003 net=0.003 yaw_rate_ratio=0.854
     lat_l: err rms=0.021 initial=0.014 final=0.027 max=0.052 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.003 yaw_rate_ratio=n/a
     lat_r: err rms=0.026 initial=0.015 final=0.025 max=0.059 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
      back: err rms=0.081 initial=0.014 final=0.098 max=0.158 closed=n/a | settle=8.940 (n=99/100) | progress=-0.001 net=-0.005 yaw_rate_ratio=0.710
      diag: err rms=0.035 initial=0.014 final=0.033 max=0.085 closed=n/a | settle=0.000 (n=100/100) | progress=0.001 net=0.002 yaw_rate_ratio=0.324
  turn_l90: err rms=0.788 initial=1.564 final=0.019 max=1.572 closed=0.988 | settle=8.460 (n=100/100) | progress=0.168 net=0.131 yaw_rate_ratio=0.648
  turn_r90: err rms=0.819 initial=1.582 final=0.024 max=1.588 closed=0.985 | settle=8.660 (n=100/100) | progress=0.167 net=0.131 yaw_rate_ratio=0.648
  move_turn: err rms=1.163 initial=1.564 final=0.840 max=1.573 closed=0.462 | settle=n/a (n=0/100) | progress=0.060 net=0.060 yaw_rate_ratio=0.446
  move_turn_r: err rms=1.023 initial=1.578 final=0.390 max=1.586 closed=0.752 | settle=11.860 (n=3/100) | progress=0.099 net=0.099 yaw_rate_ratio=0.470

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: stand, fwd, side, back, diag, turn, move_turn):
      stand: n=100 surv=1.000 (stand=1.00) | ratio=n/a achieved=0.003 tripod=0.730 | closed=n/a progress=0.006 net=-0.000 | drift=0.030 dir_err=n/a
        fwd: n=100 surv=1.000 (fwd=1.00) | ratio=1.017 achieved=0.254 tripod=0.640 | closed=n/a progress=-0.003 net=0.003 | drift=0.254 dir_err=0.022
       side: n=200 surv=1.000 (lat_l=1.00 lat_r=1.00) | ratio=1.044 achieved=0.230 tripod=0.686 | closed=n/a progress=n/a net=0.001 | drift=0.231 dir_err=0.020
       back: n=100 surv=1.000 (back=1.00) | ratio=1.017 achieved=0.224 tripod=0.682 | closed=n/a progress=-0.001 net=-0.005 | drift=0.224 dir_err=0.021
       diag: n=100 surv=1.000 (diag=1.00) | ratio=1.034 achieved=0.259 tripod=0.666 | closed=n/a progress=0.001 net=0.002 | drift=0.260 dir_err=0.038
       turn: n=200 surv=1.000 (turn_l90=1.00 turn_r90=1.00) | ratio=n/a achieved=-0.000 tripod=0.760 | closed=0.987 progress=0.167 net=0.131 | drift=0.048 dir_err=n/a
  move_turn: n=200 surv=0.995 (move_turn=0.99 move_turn_r=1.00) | ratio=0.798 achieved=0.176 tripod=0.577 | closed=0.596 progress=0.077 net=0.078 | drift=0.188 dir_err=0.098

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
      stand: action_jitter=0.115 action_fft_freq=1.298 joint_speed_mean=0.360 rod_speed_mean=0.028 contact_switch_hz=1.104 foot_slip=0.085 wz_active_frac=0.000 clock_restarts=0.000 six_down_frac=0.032 settle_six_down_s=12.000 foot_disp_end=0.111 work_per_s=403.213
        fwd: action_jitter=0.087 action_fft_freq=0.933 joint_speed_mean=0.754 rod_speed_mean=0.029 contact_switch_hz=1.556 foot_slip=0.112 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=18.774 slip_per_m=1.389 work_per_m=1472.661 cam_work_per_m=614.792
       side: action_jitter=0.069 action_fft_freq=0.973 joint_speed_mean=0.385 rod_speed_mean=0.028 contact_switch_hz=1.194 foot_slip=0.092 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=15.881 slip_per_m=1.303 work_per_m=1432.585 cam_work_per_m=263.635
       back: action_jitter=0.092 action_fft_freq=0.975 joint_speed_mean=0.712 rod_speed_mean=0.029 contact_switch_hz=1.528 foot_slip=0.109 wz_active_frac=0.038 clock_restarts=0.000 swings_per_m=20.896 slip_per_m=1.580 work_per_m=1574.585 cam_work_per_m=546.982
       diag: action_jitter=0.084 action_fft_freq=1.031 joint_speed_mean=0.572 rod_speed_mean=0.029 contact_switch_hz=1.569 foot_slip=0.108 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=18.356 slip_per_m=1.316 work_per_m=1291.842 cam_work_per_m=280.628
       turn: action_jitter=0.093 action_fft_freq=0.965 joint_speed_mean=0.690 rod_speed_mean=0.029 contact_switch_hz=1.229 foot_slip=0.088 wz_active_frac=0.717 clock_restarts=0.000 swings_per_rad=28.492 slip_per_rad=2.208 work_per_rad=2427.549 cam_work_per_rad=808.660
  move_turn: action_jitter=0.081 action_fft_freq=0.941 joint_speed_mean=0.677 rod_speed_mean=0.028 contact_switch_hz=1.583 foot_slip=0.118 wz_active_frac=1.000 clock_restarts=0.000 swings_per_rad=60.766 slip_per_rad=4.734 work_per_rad=4888.549 cam_work_per_rad=2022.019

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
      stand: support_feet=3.376 steps_per_s=2.167 gait_hz=0.360 clock_hz=0.000 cadence_ratio=n/a
        fwd: steps_per_m=10.696 support_feet=3.152 taps_per_m=7.951 swing_apex=0.116 step_rate_hz=2.750 gait_hz=0.391 clock_hz=0.357 cadence_ratio=1.092 hz_speed_slope=-0.310 offset_R=0.113
       side: steps_per_m=10.792 support_feet=3.218 taps_per_m=5.122 swing_apex=0.098 step_rate_hz=2.458 gait_hz=0.378 clock_hz=0.314 cadence_ratio=1.204 hz_speed_slope=2.954 offset_R=0.018
       back: steps_per_m=12.021 support_feet=3.173 taps_per_m=8.766 swing_apex=0.107 step_rate_hz=2.667 gait_hz=0.403 clock_hz=0.317 cadence_ratio=1.275 hz_speed_slope=-0.380 offset_R=0.063
       diag: steps_per_m=11.529 support_feet=3.110 taps_per_m=6.845 swing_apex=0.091 step_rate_hz=2.958 gait_hz=0.439 clock_hz=0.357 cadence_ratio=1.230 hz_speed_slope=0.353 offset_R=0.096
       turn: steps_per_rad=19.457 support_feet=3.272 taps_per_rad=8.771 swing_apex=0.136 step_rate_hz=2.542 gait_hz=0.409 clock_hz=0.357 cadence_ratio=1.169 steps_per_rad_pre=14.450 steps_per_rad_post=5.547 offset_R=0.112
  move_turn: steps_per_rad=34.651 support_feet=3.127 taps_per_rad=26.045 swing_apex=0.104 step_rate_hz=2.667 gait_hz=0.366 clock_hz=0.500 cadence_ratio=0.732 hz_speed_slope=-0.170 offset_R=0.115

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
