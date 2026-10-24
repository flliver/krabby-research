```
=== crab-hex gait eval ===
scenario   : dirsweep__omni_body  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-21_23-06-12_seed19_X06_p2a_A15pB/model_19994.pt
episodes   : 1600  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7501702463404889  p25=0.7287852749093982  p75=0.7684206546315618
tippy_tap_fraction median=0.4268292682926829
slip_ratio         median=0.21068479496072956
tracking_ratio     median=1.0674208452965028  (achieved/commanded vx, walking holds; n=1600)
schedule_completion_rate=1.0
terminations={'schedule_complete': 1600}
facing_error_rms   median=0.03698297827771027  (rad, whole episode; n=1600)

by hold:
    settle: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.7802266621404762 (n=1577)
      d000: cmd 0.22 -> achieved 0.235 m/s | tripod median=0.7043811225376069 (n=100)
      d022: cmd 0.22 -> achieved 0.229 m/s | tripod median=0.7496840348870131 (n=100)
      d045: cmd 0.22 -> achieved 0.226 m/s | tripod median=0.7631587816697765 (n=100)
      d068: cmd 0.22 -> achieved 0.233 m/s | tripod median=0.726653486333287 (n=100)
      d090: cmd 0.22 -> achieved 0.234 m/s | tripod median=0.7266277645142256 (n=100)
      d112: cmd 0.22 -> achieved 0.233 m/s | tripod median=0.7233564132725512 (n=100)
      d135: cmd 0.22 -> achieved 0.232 m/s | tripod median=0.7551112763207817 (n=100)
      d158: cmd 0.22 -> achieved 0.234 m/s | tripod median=0.7743108895973978 (n=100)
      d180: cmd 0.22 -> achieved 0.234 m/s | tripod median=0.7490265007623315 (n=100)
      d202: cmd 0.22 -> achieved 0.239 m/s | tripod median=0.7597286999354748 (n=100)
      d225: cmd 0.22 -> achieved 0.236 m/s | tripod median=0.7594005216096238 (n=100)
      d248: cmd 0.22 -> achieved 0.237 m/s | tripod median=0.7680065517377586 (n=100)
      d270: cmd 0.22 -> achieved 0.235 m/s | tripod median=0.7958770715795547 (n=100)
      d292: cmd 0.22 -> achieved 0.239 m/s | tripod median=0.7503865462666512 (n=100)
      d315: cmd 0.22 -> achieved 0.237 m/s | tripod median=0.7388269056966592 (n=100)
      d338: cmd 0.22 -> achieved 0.242 m/s | tripod median=0.724882486026423 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.018 initial=0.006 final=0.015 max=0.034 closed=n/a | settle=0.000 (n=1600/1600) | progress=n/a net=0.001 yaw_rate_ratio=n/a
      d000: err rms=0.057 initial=0.013 final=0.088 max=0.109 closed=n/a | settle=0.000 (n=100/100) | progress=0.000 net=-0.003 yaw_rate_ratio=0.225
      d022: err rms=0.071 initial=0.016 final=0.096 max=0.137 closed=n/a | settle=0.000 (n=97/100) | progress=-0.002 net=-0.003 yaw_rate_ratio=0.376
      d045: err rms=0.045 initial=0.016 final=0.052 max=0.086 closed=n/a | settle=0.000 (n=100/100) | progress=0.002 net=-0.000 yaw_rate_ratio=0.114
      d068: err rms=0.022 initial=0.016 final=0.015 max=0.053 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.002 yaw_rate_ratio=n/a
      d090: err rms=0.025 initial=0.014 final=0.028 max=0.054 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a
      d112: err rms=0.037 initial=0.016 final=0.042 max=0.073 closed=n/a | settle=0.000 (n=100/100) | progress=-0.012 net=-0.001 yaw_rate_ratio=0.287
      d135: err rms=0.031 initial=0.015 final=0.021 max=0.072 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.002 yaw_rate_ratio=n/a
      d158: err rms=0.033 initial=0.018 final=0.027 max=0.074 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.002 yaw_rate_ratio=n/a
      d180: err rms=0.036 initial=0.014 final=0.037 max=0.079 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.004 yaw_rate_ratio=n/a
      d202: err rms=0.052 initial=0.015 final=0.045 max=0.095 closed=n/a | settle=0.000 (n=100/100) | progress=-0.003 net=0.004 yaw_rate_ratio=0.338
      d225: err rms=0.057 initial=0.014 final=0.055 max=0.103 closed=n/a | settle=0.000 (n=100/100) | progress=-0.001 net=0.005 yaw_rate_ratio=0.313
      d248: err rms=0.025 initial=0.017 final=0.017 max=0.058 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.002 yaw_rate_ratio=n/a
      d270: err rms=0.023 initial=0.014 final=0.017 max=0.047 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
      d292: err rms=0.040 initial=0.018 final=0.041 max=0.078 closed=n/a | settle=0.000 (n=100/100) | progress=0.001 net=-0.001 yaw_rate_ratio=0.334
      d315: err rms=0.078 initial=0.013 final=0.081 max=0.122 closed=n/a | settle=0.000 (n=100/100) | progress=-0.000 net=-0.003 yaw_rate_ratio=0.287
      d338: err rms=0.061 initial=0.012 final=0.068 max=0.110 closed=n/a | settle=0.000 (n=100/100) | progress=-0.002 net=-0.003 yaw_rate_ratio=0.291

body yaw by hold (window [4 s, hold_s); medians; net = net yaw rate rad/s, ratio = net / wz_cmd SIGNED (the gated turn number), wz_ratio = mean wz / wz_cmd; ripple = std of the 1-s mean wz; drift = |mean wz| on wz = 0 holds; planar = xy drift m/s on pure turns; body = along-bearing speed / |v_cmd|, cross = mean |perp| m/s):
          settle: net=n/a ratio=n/a wz_ratio=n/a ripple=n/a | drift=n/a planar=n/a | body=n/a cross=n/a (n=0)
            d000: net=0.009 ratio=n/a wz_ratio=n/a ripple=0.021 | drift=0.009 planar=n/a | body=1.075 cross=0.026 (n=100)
            d022: net=0.007 ratio=n/a wz_ratio=n/a ripple=0.023 | drift=0.007 planar=n/a | body=1.035 cross=0.029 (n=100)
            d045: net=0.004 ratio=n/a wz_ratio=n/a ripple=0.022 | drift=0.005 planar=n/a | body=1.029 cross=0.026 (n=100)
            d068: net=0.000 ratio=n/a wz_ratio=n/a ripple=0.017 | drift=0.003 planar=n/a | body=1.066 cross=0.025 (n=100)
            d090: net=0.002 ratio=n/a wz_ratio=n/a ripple=0.013 | drift=0.003 planar=n/a | body=1.071 cross=0.022 (n=100)
            d112: net=0.001 ratio=n/a wz_ratio=n/a ripple=0.017 | drift=0.002 planar=n/a | body=1.064 cross=0.025 (n=100)
            d135: net=-0.003 ratio=n/a wz_ratio=n/a ripple=0.023 | drift=0.005 planar=n/a | body=1.062 cross=0.028 (n=100)
            d158: net=0.000 ratio=n/a wz_ratio=n/a ripple=0.026 | drift=0.004 planar=n/a | body=1.068 cross=0.028 (n=100)
            d180: net=-0.003 ratio=n/a wz_ratio=n/a ripple=0.023 | drift=0.007 planar=n/a | body=1.062 cross=0.023 (n=100)
            d202: net=-0.002 ratio=n/a wz_ratio=n/a ripple=0.020 | drift=0.006 planar=n/a | body=1.092 cross=0.024 (n=100)
            d225: net=-0.001 ratio=n/a wz_ratio=n/a ripple=0.019 | drift=0.003 planar=n/a | body=1.073 cross=0.032 (n=100)
            d248: net=-0.000 ratio=n/a wz_ratio=n/a ripple=0.020 | drift=0.003 planar=n/a | body=1.077 cross=0.035 (n=100)
            d270: net=0.000 ratio=n/a wz_ratio=n/a ripple=0.011 | drift=0.002 planar=n/a | body=1.068 cross=0.022 (n=100)
            d292: net=-0.000 ratio=n/a wz_ratio=n/a ripple=0.021 | drift=0.003 planar=n/a | body=1.090 cross=0.030 (n=100)
            d315: net=0.002 ratio=n/a wz_ratio=n/a ripple=0.018 | drift=0.003 planar=n/a | body=1.080 cross=0.031 (n=100)
            d338: net=0.003 ratio=n/a wz_ratio=n/a ripple=0.021 | drift=0.004 planar=n/a | body=1.101 cross=0.024 (n=100)

by category (16 waves, 1600 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: n/a):
       d000: n=100 surv=1.000 (d000=1.00) | ratio=1.070 achieved=0.235 tripod=0.704 | closed=n/a progress=0.000 net=-0.003 | drift=0.234 dir_err=0.067
       d022: n=100 surv=1.000 (d022=1.00) | ratio=1.039 achieved=0.229 tripod=0.750 | closed=n/a progress=-0.002 net=-0.003 | drift=0.229 dir_err=0.073
       d045: n=100 surv=1.000 (d045=1.00) | ratio=1.025 achieved=0.226 tripod=0.763 | closed=n/a progress=0.002 net=-0.000 | drift=0.228 dir_err=0.014
       d068: n=100 surv=1.000 (d068=1.00) | ratio=1.060 achieved=0.233 tripod=0.727 | closed=n/a progress=n/a net=0.002 | drift=0.234 dir_err=0.017
       d090: n=100 surv=1.000 (d090=1.00) | ratio=1.065 achieved=0.234 tripod=0.727 | closed=n/a progress=n/a net=0.000 | drift=0.234 dir_err=0.011
       d112: n=100 surv=1.000 (d112=1.00) | ratio=1.058 achieved=0.233 tripod=0.723 | closed=n/a progress=-0.012 net=-0.001 | drift=0.234 dir_err=0.012
       d135: n=100 surv=1.000 (d135=1.00) | ratio=1.056 achieved=0.232 tripod=0.755 | closed=n/a progress=n/a net=0.002 | drift=0.235 dir_err=0.023
       d158: n=100 surv=1.000 (d158=1.00) | ratio=1.064 achieved=0.234 tripod=0.774 | closed=n/a progress=n/a net=0.002 | drift=0.235 dir_err=0.062
       d180: n=100 surv=1.000 (d180=1.00) | ratio=1.062 achieved=0.234 tripod=0.749 | closed=n/a progress=n/a net=0.004 | drift=0.233 dir_err=0.044
       d202: n=100 surv=1.000 (d202=1.00) | ratio=1.087 achieved=0.239 tripod=0.760 | closed=n/a progress=-0.003 net=0.004 | drift=0.238 dir_err=0.026
       d225: n=100 surv=1.000 (d225=1.00) | ratio=1.073 achieved=0.236 tripod=0.759 | closed=n/a progress=-0.001 net=0.005 | drift=0.236 dir_err=0.043
       d248: n=100 surv=1.000 (d248=1.00) | ratio=1.077 achieved=0.237 tripod=0.768 | closed=n/a progress=n/a net=0.002 | drift=0.238 dir_err=0.049
       d270: n=100 surv=1.000 (d270=1.00) | ratio=1.068 achieved=0.235 tripod=0.796 | closed=n/a progress=n/a net=0.001 | drift=0.234 dir_err=0.018
       d292: n=100 surv=1.000 (d292=1.00) | ratio=1.086 achieved=0.239 tripod=0.750 | closed=n/a progress=0.001 net=-0.001 | drift=0.239 dir_err=0.032
       d315: n=100 surv=1.000 (d315=1.00) | ratio=1.079 achieved=0.237 tripod=0.739 | closed=n/a progress=-0.000 net=-0.003 | drift=0.237 dir_err=0.042
       d338: n=100 surv=1.000 (d338=1.00) | ratio=1.101 achieved=0.242 tripod=0.725 | closed=n/a progress=-0.002 net=-0.003 | drift=0.240 dir_err=0.036

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
       d000: action_jitter=0.174 action_fft_freq=1.331 joint_speed_mean=0.727 rod_speed_mean=0.029 contact_switch_hz=1.819 foot_slip=0.096 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=23.646 slip_per_m=1.222 work_per_m=1128.728 cam_work_per_m=413.247
       d022: action_jitter=0.173 action_fft_freq=1.338 joint_speed_mean=0.700 rod_speed_mean=0.029 contact_switch_hz=1.778 foot_slip=0.094 wz_active_frac=0.012 clock_restarts=0.000 swings_per_m=23.444 slip_per_m=1.262 work_per_m=1134.949 cam_work_per_m=411.290
       d045: action_jitter=0.172 action_fft_freq=1.448 joint_speed_mean=0.583 rod_speed_mean=0.029 contact_switch_hz=1.806 foot_slip=0.077 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=24.179 slip_per_m=1.047 work_per_m=1040.075 cam_work_per_m=265.672
       d068: action_jitter=0.151 action_fft_freq=1.485 joint_speed_mean=0.448 rod_speed_mean=0.029 contact_switch_hz=1.757 foot_slip=0.078 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=22.773 slip_per_m=1.030 work_per_m=1017.608 cam_work_per_m=244.143
       d090: action_jitter=0.137 action_fft_freq=1.412 joint_speed_mean=0.340 rod_speed_mean=0.029 contact_switch_hz=1.521 foot_slip=0.074 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=19.714 slip_per_m=0.977 work_per_m=993.974 cam_work_per_m=246.410
       d112: action_jitter=0.149 action_fft_freq=1.479 joint_speed_mean=0.432 rod_speed_mean=0.029 contact_switch_hz=1.750 foot_slip=0.078 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=22.628 slip_per_m=1.033 work_per_m=997.460 cam_work_per_m=275.413
       d135: action_jitter=0.157 action_fft_freq=1.387 joint_speed_mean=0.575 rod_speed_mean=0.029 contact_switch_hz=1.722 foot_slip=0.082 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=22.395 slip_per_m=1.085 work_per_m=1006.550 cam_work_per_m=303.110
       d158: action_jitter=0.161 action_fft_freq=1.316 joint_speed_mean=0.683 rod_speed_mean=0.029 contact_switch_hz=1.722 foot_slip=0.098 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=21.795 slip_per_m=1.281 work_per_m=1101.497 cam_work_per_m=410.533
       d180: action_jitter=0.164 action_fft_freq=1.277 joint_speed_mean=0.719 rod_speed_mean=0.029 contact_switch_hz=1.806 foot_slip=0.101 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=23.404 slip_per_m=1.290 work_per_m=1127.679 cam_work_per_m=399.471
       d202: action_jitter=0.165 action_fft_freq=1.298 joint_speed_mean=0.691 rod_speed_mean=0.029 contact_switch_hz=1.875 foot_slip=0.099 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=23.739 slip_per_m=1.247 work_per_m=1114.860 cam_work_per_m=366.985
       d225: action_jitter=0.154 action_fft_freq=1.348 joint_speed_mean=0.595 rod_speed_mean=0.029 contact_switch_hz=1.889 foot_slip=0.093 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=24.207 slip_per_m=1.192 work_per_m=1079.962 cam_work_per_m=305.546
       d248: action_jitter=0.134 action_fft_freq=1.341 joint_speed_mean=0.460 rod_speed_mean=0.029 contact_switch_hz=1.708 foot_slip=0.080 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=21.849 slip_per_m=1.045 work_per_m=1059.113 cam_work_per_m=207.720
       d270: action_jitter=0.134 action_fft_freq=1.431 joint_speed_mean=0.349 rod_speed_mean=0.029 contact_switch_hz=1.611 foot_slip=0.074 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=20.772 slip_per_m=0.989 work_per_m=1040.422 cam_work_per_m=178.849
       d292: action_jitter=0.141 action_fft_freq=1.365 joint_speed_mean=0.459 rod_speed_mean=0.029 contact_switch_hz=1.778 foot_slip=0.083 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=22.425 slip_per_m=1.082 work_per_m=1060.660 cam_work_per_m=214.831
       d315: action_jitter=0.158 action_fft_freq=1.362 joint_speed_mean=0.598 rod_speed_mean=0.029 contact_switch_hz=1.812 foot_slip=0.087 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=23.329 slip_per_m=1.119 work_per_m=1054.116 cam_work_per_m=290.732
       d338: action_jitter=0.171 action_fft_freq=1.326 joint_speed_mean=0.691 rod_speed_mean=0.029 contact_switch_hz=1.889 foot_slip=0.093 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=23.885 slip_per_m=1.170 work_per_m=1095.591 cam_work_per_m=374.120

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
       d000: steps_per_m=10.922 support_feet=2.957 taps_per_m=12.667 swing_apex=0.123 step_rate_hz=2.542 gait_hz=0.397 clock_hz=0.314 cadence_ratio=1.262 hz_speed_slope=-0.661 offset_R=0.091
       d022: steps_per_m=11.730 support_feet=3.028 taps_per_m=11.731 swing_apex=0.118 step_rate_hz=2.667 gait_hz=0.427 clock_hz=0.315 cadence_ratio=1.359 hz_speed_slope=-0.856 offset_R=0.146
       d045: steps_per_m=13.863 support_feet=3.054 taps_per_m=10.069 swing_apex=0.097 step_rate_hz=3.083 gait_hz=0.519 clock_hz=0.314 cadence_ratio=1.651 hz_speed_slope=0.614 offset_R=0.078
       d068: steps_per_m=14.292 support_feet=3.072 taps_per_m=8.467 swing_apex=0.082 step_rate_hz=3.333 gait_hz=0.555 clock_hz=0.314 cadence_ratio=1.767 hz_speed_slope=1.352 offset_R=0.112
       d090: steps_per_m=12.222 support_feet=3.075 taps_per_m=7.192 swing_apex=0.089 step_rate_hz=2.833 gait_hz=0.482 clock_hz=0.314 cadence_ratio=1.534 hz_speed_slope=-0.314 offset_R=0.064
       d112: steps_per_m=13.496 support_feet=3.072 taps_per_m=9.034 swing_apex=0.083 step_rate_hz=3.083 gait_hz=0.519 clock_hz=0.314 cadence_ratio=1.651 hz_speed_slope=2.372 offset_R=0.069
       d135: steps_per_m=13.304 support_feet=3.048 taps_per_m=9.092 swing_apex=0.096 step_rate_hz=3.083 gait_hz=0.507 clock_hz=0.314 cadence_ratio=1.612 hz_speed_slope=1.273 offset_R=0.114
       d158: steps_per_m=11.459 support_feet=3.041 taps_per_m=10.015 swing_apex=0.122 step_rate_hz=2.667 gait_hz=0.439 clock_hz=0.314 cadence_ratio=1.398 hz_speed_slope=-1.077 offset_R=0.017
       d180: steps_per_m=11.047 support_feet=2.974 taps_per_m=12.345 swing_apex=0.124 step_rate_hz=2.542 gait_hz=0.403 clock_hz=0.314 cadence_ratio=1.282 hz_speed_slope=-0.248 offset_R=0.077
       d202: steps_per_m=11.383 support_feet=2.992 taps_per_m=12.324 swing_apex=0.117 step_rate_hz=2.667 gait_hz=0.427 clock_hz=0.314 cadence_ratio=1.359 hz_speed_slope=-0.395 offset_R=0.056
       d225: steps_per_m=12.508 support_feet=3.000 taps_per_m=11.387 swing_apex=0.102 step_rate_hz=2.917 gait_hz=0.476 clock_hz=0.314 cadence_ratio=1.515 hz_speed_slope=-0.024 offset_R=0.084
       d248: steps_per_m=13.708 support_feet=3.051 taps_per_m=7.775 swing_apex=0.082 step_rate_hz=3.167 gait_hz=0.531 clock_hz=0.314 cadence_ratio=1.690 hz_speed_slope=-0.365 offset_R=0.026
       d270: steps_per_m=13.483 support_feet=3.095 taps_per_m=7.173 swing_apex=0.079 step_rate_hz=3.167 gait_hz=0.537 clock_hz=0.314 cadence_ratio=1.709 hz_speed_slope=2.801 offset_R=0.090
       d292: steps_per_m=13.473 support_feet=3.068 taps_per_m=9.014 swing_apex=0.086 step_rate_hz=3.167 gait_hz=0.513 clock_hz=0.314 cadence_ratio=1.631 hz_speed_slope=5.236 offset_R=0.207
       d315: steps_per_m=12.648 support_feet=3.010 taps_per_m=10.608 swing_apex=0.104 step_rate_hz=2.917 gait_hz=0.470 clock_hz=0.314 cadence_ratio=1.495 hz_speed_slope=0.016 offset_R=0.104
       d338: steps_per_m=11.601 support_feet=2.999 taps_per_m=12.235 swing_apex=0.115 step_rate_hz=2.750 gait_hz=0.421 clock_hz=0.314 cadence_ratio=1.340 hz_speed_slope=-0.009 offset_R=0.059

body yaw by category (own hold, window [4 s, hold_s); medians pooled over both hands; per-hand values are in aggregate.suite.by_category[c].by_hand):
       d000: net_yaw_rate=0.009 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.021 heading_drift=0.009 planar_drift=n/a body_ratio=1.075 cross_body=0.026 adapter_clipped_frac=n/a
       d022: net_yaw_rate=0.007 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.023 heading_drift=0.007 planar_drift=n/a body_ratio=1.035 cross_body=0.029 adapter_clipped_frac=n/a
       d045: net_yaw_rate=0.004 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.022 heading_drift=0.005 planar_drift=n/a body_ratio=1.029 cross_body=0.026 adapter_clipped_frac=n/a
       d068: net_yaw_rate=0.000 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.017 heading_drift=0.003 planar_drift=n/a body_ratio=1.066 cross_body=0.025 adapter_clipped_frac=n/a
       d090: net_yaw_rate=0.002 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.013 heading_drift=0.003 planar_drift=n/a body_ratio=1.071 cross_body=0.022 adapter_clipped_frac=n/a
       d112: net_yaw_rate=0.001 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.017 heading_drift=0.002 planar_drift=n/a body_ratio=1.064 cross_body=0.025 adapter_clipped_frac=n/a
       d135: net_yaw_rate=-0.003 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.023 heading_drift=0.005 planar_drift=n/a body_ratio=1.062 cross_body=0.028 adapter_clipped_frac=n/a
       d158: net_yaw_rate=0.000 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.026 heading_drift=0.004 planar_drift=n/a body_ratio=1.068 cross_body=0.028 adapter_clipped_frac=n/a
       d180: net_yaw_rate=-0.003 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.023 heading_drift=0.007 planar_drift=n/a body_ratio=1.062 cross_body=0.023 adapter_clipped_frac=n/a
       d202: net_yaw_rate=-0.002 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.020 heading_drift=0.006 planar_drift=n/a body_ratio=1.092 cross_body=0.024 adapter_clipped_frac=n/a
       d225: net_yaw_rate=-0.001 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.019 heading_drift=0.003 planar_drift=n/a body_ratio=1.073 cross_body=0.032 adapter_clipped_frac=n/a
       d248: net_yaw_rate=-0.000 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.020 heading_drift=0.003 planar_drift=n/a body_ratio=1.077 cross_body=0.035 adapter_clipped_frac=n/a
       d270: net_yaw_rate=0.000 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.011 heading_drift=0.002 planar_drift=n/a body_ratio=1.068 cross_body=0.022 adapter_clipped_frac=n/a
       d292: net_yaw_rate=-0.000 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.021 heading_drift=0.003 planar_drift=n/a body_ratio=1.090 cross_body=0.030 adapter_clipped_frac=n/a
       d315: net_yaw_rate=0.002 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.018 heading_drift=0.003 planar_drift=n/a body_ratio=1.080 cross_body=0.031 adapter_clipped_frac=n/a
       d338: net_yaw_rate=0.003 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.021 heading_drift=0.004 planar_drift=n/a body_ratio=1.101 cross_body=0.024 adapter_clipped_frac=n/a

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
