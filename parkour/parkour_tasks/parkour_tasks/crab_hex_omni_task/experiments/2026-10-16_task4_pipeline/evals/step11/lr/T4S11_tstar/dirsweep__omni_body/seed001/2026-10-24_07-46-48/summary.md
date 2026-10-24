```
=== crab-hex gait eval ===
scenario   : dirsweep__omni_body  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt
episodes   : 1600  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7104998299322143  p25=0.6759821130770151  p75=0.7439373445036593
tippy_tap_fraction median=0.4418604651162791
slip_ratio         median=0.22901279452963486
tracking_ratio     median=1.0391653200816529  (achieved/commanded vx, walking holds; n=1600)
schedule_completion_rate=1.0
terminations={'schedule_complete': 1600}
facing_error_rms   median=0.03550529185848708  (rad, whole episode; n=1600)

by hold:
    settle: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.767840136461866 (n=1571)
      d000: cmd 0.22 -> achieved 0.228 m/s | tripod median=0.7042304640667241 (n=100)
      d022: cmd 0.22 -> achieved 0.228 m/s | tripod median=0.7440473415213188 (n=100)
      d045: cmd 0.22 -> achieved 0.234 m/s | tripod median=0.7033811101668203 (n=100)
      d068: cmd 0.22 -> achieved 0.236 m/s | tripod median=0.6654120235525407 (n=100)
      d090: cmd 0.22 -> achieved 0.235 m/s | tripod median=0.6786385199900098 (n=100)
      d112: cmd 0.22 -> achieved 0.234 m/s | tripod median=0.6658229491328876 (n=100)
      d135: cmd 0.22 -> achieved 0.230 m/s | tripod median=0.64445404815679 (n=100)
      d158: cmd 0.22 -> achieved 0.218 m/s | tripod median=0.7517237263005745 (n=100)
      d180: cmd 0.22 -> achieved 0.218 m/s | tripod median=0.7027515876917265 (n=100)
      d202: cmd 0.22 -> achieved 0.218 m/s | tripod median=0.6684553069581707 (n=100)
      d225: cmd 0.22 -> achieved 0.223 m/s | tripod median=0.7170784746145022 (n=100)
      d248: cmd 0.22 -> achieved 0.229 m/s | tripod median=0.7494298632835006 (n=100)
      d270: cmd 0.22 -> achieved 0.232 m/s | tripod median=0.7612871601155936 (n=100)
      d292: cmd 0.22 -> achieved 0.229 m/s | tripod median=0.740098965367967 (n=100)
      d315: cmd 0.22 -> achieved 0.228 m/s | tripod median=0.7436872986399232 (n=100)
      d338: cmd 0.22 -> achieved 0.227 m/s | tripod median=0.6685251130380292 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.014 initial=0.008 final=0.012 max=0.028 closed=n/a | settle=0.000 (n=1600/1600) | progress=n/a net=0.002 yaw_rate_ratio=n/a
      d000: err rms=0.044 initial=0.013 final=0.033 max=0.103 closed=n/a | settle=0.000 (n=100/100) | progress=-0.001 net=0.001 yaw_rate_ratio=0.341
      d022: err rms=0.049 initial=0.013 final=0.036 max=0.095 closed=n/a | settle=0.000 (n=100/100) | progress=0.001 net=0.002 yaw_rate_ratio=0.506
      d045: err rms=0.027 initial=0.012 final=0.015 max=0.061 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
      d068: err rms=0.037 initial=0.013 final=0.055 max=0.083 closed=n/a | settle=0.000 (n=100/100) | progress=-0.001 net=0.003 yaw_rate_ratio=0.487
      d090: err rms=0.019 initial=0.015 final=0.014 max=0.045 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
      d112: err rms=0.043 initial=0.013 final=0.058 max=0.096 closed=n/a | settle=0.000 (n=100/100) | progress=-0.003 net=0.000 yaw_rate_ratio=0.328
      d135: err rms=0.031 initial=0.013 final=0.025 max=0.074 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
      d158: err rms=0.070 initial=0.011 final=0.106 max=0.135 closed=n/a | settle=0.000 (n=94/100) | progress=-0.005 net=-0.003 yaw_rate_ratio=0.369
      d180: err rms=0.060 initial=0.011 final=0.083 max=0.129 closed=n/a | settle=0.000 (n=100/100) | progress=-0.002 net=0.001 yaw_rate_ratio=0.534
      d202: err rms=0.041 initial=0.013 final=0.029 max=0.099 closed=n/a | settle=0.000 (n=100/100) | progress=-0.003 net=0.001 yaw_rate_ratio=0.488
      d225: err rms=0.030 initial=0.015 final=0.021 max=0.073 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
      d248: err rms=0.035 initial=0.013 final=0.033 max=0.078 closed=n/a | settle=0.000 (n=100/100) | progress=-0.000 net=0.001 yaw_rate_ratio=0.404
      d270: err rms=0.029 initial=0.010 final=0.030 max=0.067 closed=n/a | settle=0.000 (n=100/100) | progress=-0.004 net=0.001 yaw_rate_ratio=0.155
      d292: err rms=0.060 initial=0.011 final=0.084 max=0.118 closed=n/a | settle=0.000 (n=99/100) | progress=-0.001 net=-0.002 yaw_rate_ratio=0.350
      d315: err rms=0.054 initial=0.014 final=0.052 max=0.105 closed=n/a | settle=0.000 (n=100/100) | progress=-0.002 net=0.000 yaw_rate_ratio=0.333
      d338: err rms=0.034 initial=0.013 final=0.018 max=0.083 closed=n/a | settle=0.000 (n=100/100) | progress=0.012 net=0.001 yaw_rate_ratio=0.689

body yaw by hold (window [4 s, hold_s); medians; net = net yaw rate rad/s, ratio = net / wz_cmd SIGNED (the gated turn number), wz_ratio = mean wz / wz_cmd; ripple = std of the 1-s mean wz; drift = |mean wz| on wz = 0 holds; planar = xy drift m/s on pure turns; body = along-bearing speed / |v_cmd|, cross = mean |perp| m/s):
          settle: net=n/a ratio=n/a wz_ratio=n/a ripple=n/a | drift=n/a planar=n/a | body=n/a cross=n/a (n=0)
            d000: net=-0.004 ratio=n/a wz_ratio=n/a ripple=0.043 | drift=0.006 planar=n/a | body=1.033 cross=0.025 (n=100)
            d022: net=-0.002 ratio=n/a wz_ratio=n/a ripple=0.025 | drift=0.005 planar=n/a | body=1.036 cross=0.024 (n=100)
            d045: net=-0.002 ratio=n/a wz_ratio=n/a ripple=0.026 | drift=0.004 planar=n/a | body=1.065 cross=0.036 (n=100)
            d068: net=-0.007 ratio=n/a wz_ratio=n/a ripple=0.027 | drift=0.007 planar=n/a | body=1.077 cross=0.033 (n=100)
            d090: net=-0.000 ratio=n/a wz_ratio=n/a ripple=0.015 | drift=0.002 planar=n/a | body=1.069 cross=0.025 (n=100)
            d112: net=0.005 ratio=n/a wz_ratio=n/a ripple=0.029 | drift=0.006 planar=n/a | body=1.070 cross=0.034 (n=100)
            d135: net=0.001 ratio=n/a wz_ratio=n/a ripple=0.023 | drift=0.004 planar=n/a | body=1.047 cross=0.032 (n=100)
            d158: net=0.011 ratio=n/a wz_ratio=n/a ripple=0.022 | drift=0.012 planar=n/a | body=0.984 cross=0.024 (n=100)
            d180: net=0.008 ratio=n/a wz_ratio=n/a ripple=0.030 | drift=0.008 planar=n/a | body=0.985 cross=0.026 (n=100)
            d202: net=0.001 ratio=n/a wz_ratio=n/a ripple=0.045 | drift=0.003 planar=n/a | body=0.997 cross=0.031 (n=100)
            d225: net=0.003 ratio=n/a wz_ratio=n/a ripple=0.038 | drift=0.004 planar=n/a | body=1.013 cross=0.032 (n=100)
            d248: net=-0.001 ratio=n/a wz_ratio=n/a ripple=0.025 | drift=0.003 planar=n/a | body=1.048 cross=0.030 (n=100)
            d270: net=0.003 ratio=n/a wz_ratio=n/a ripple=0.021 | drift=0.004 planar=n/a | body=1.060 cross=0.027 (n=100)
            d292: net=0.006 ratio=n/a wz_ratio=n/a ripple=0.041 | drift=0.006 planar=n/a | body=1.046 cross=0.030 (n=100)
            d315: net=0.001 ratio=n/a wz_ratio=n/a ripple=0.036 | drift=0.004 planar=n/a | body=1.044 cross=0.028 (n=100)
            d338: net=-0.003 ratio=n/a wz_ratio=n/a ripple=0.038 | drift=0.004 planar=n/a | body=1.036 cross=0.030 (n=100)

by category (16 waves, 1600 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: n/a):
       d000: n=100 surv=1.000 (d000=1.00) | ratio=1.036 achieved=0.228 tripod=0.704 | closed=n/a progress=-0.001 net=0.001 | drift=0.228 dir_err=0.011
       d022: n=100 surv=1.000 (d022=1.00) | ratio=1.035 achieved=0.228 tripod=0.744 | closed=n/a progress=0.001 net=0.002 | drift=0.228 dir_err=0.019
       d045: n=100 surv=1.000 (d045=1.00) | ratio=1.061 achieved=0.234 tripod=0.703 | closed=n/a progress=n/a net=0.001 | drift=0.236 dir_err=0.076
       d068: n=100 surv=1.000 (d068=1.00) | ratio=1.072 achieved=0.236 tripod=0.665 | closed=n/a progress=-0.001 net=0.003 | drift=0.238 dir_err=0.041
       d090: n=100 surv=1.000 (d090=1.00) | ratio=1.068 achieved=0.235 tripod=0.679 | closed=n/a progress=n/a net=0.001 | drift=0.236 dir_err=0.010
       d112: n=100 surv=1.000 (d112=1.00) | ratio=1.063 achieved=0.234 tripod=0.666 | closed=n/a progress=-0.003 net=0.000 | drift=0.237 dir_err=0.047
       d135: n=100 surv=1.000 (d135=1.00) | ratio=1.044 achieved=0.230 tripod=0.644 | closed=n/a progress=n/a net=0.001 | drift=0.232 dir_err=0.067
       d158: n=100 surv=1.000 (d158=1.00) | ratio=0.990 achieved=0.218 tripod=0.752 | closed=n/a progress=-0.005 net=-0.003 | drift=0.219 dir_err=0.028
       d180: n=100 surv=1.000 (d180=1.00) | ratio=0.990 achieved=0.218 tripod=0.703 | closed=n/a progress=-0.002 net=0.001 | drift=0.218 dir_err=0.020
       d202: n=100 surv=1.000 (d202=1.00) | ratio=0.990 achieved=0.218 tripod=0.668 | closed=n/a progress=-0.003 net=0.001 | drift=0.219 dir_err=0.020
       d225: n=100 surv=1.000 (d225=1.00) | ratio=1.011 achieved=0.223 tripod=0.717 | closed=n/a progress=n/a net=0.001 | drift=0.224 dir_err=0.036
       d248: n=100 surv=1.000 (d248=1.00) | ratio=1.042 achieved=0.229 tripod=0.749 | closed=n/a progress=-0.000 net=0.001 | drift=0.231 dir_err=0.051
       d270: n=100 surv=1.000 (d270=1.00) | ratio=1.054 achieved=0.232 tripod=0.761 | closed=n/a progress=-0.004 net=0.001 | drift=0.232 dir_err=0.034
       d292: n=100 surv=1.000 (d292=1.00) | ratio=1.042 achieved=0.229 tripod=0.740 | closed=n/a progress=-0.001 net=-0.002 | drift=0.230 dir_err=0.010
       d315: n=100 surv=1.000 (d315=1.00) | ratio=1.037 achieved=0.228 tripod=0.744 | closed=n/a progress=-0.002 net=0.000 | drift=0.229 dir_err=0.028
       d338: n=100 surv=1.000 (d338=1.00) | ratio=1.033 achieved=0.227 tripod=0.669 | closed=n/a progress=0.012 net=0.001 | drift=0.228 dir_err=0.040

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
       d000: action_jitter=0.145 action_fft_freq=1.239 joint_speed_mean=0.723 rod_speed_mean=0.029 contact_switch_hz=1.847 foot_slip=0.088 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=24.593 slip_per_m=1.152 work_per_m=1144.847 cam_work_per_m=395.713
       d022: action_jitter=0.152 action_fft_freq=1.355 joint_speed_mean=0.677 rod_speed_mean=0.029 contact_switch_hz=1.917 foot_slip=0.084 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=25.637 slip_per_m=1.094 work_per_m=1108.014 cam_work_per_m=304.522
       d045: action_jitter=0.131 action_fft_freq=1.342 joint_speed_mean=0.562 rod_speed_mean=0.029 contact_switch_hz=1.736 foot_slip=0.084 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=22.464 slip_per_m=1.050 work_per_m=1051.714 cam_work_per_m=223.887
       d068: action_jitter=0.116 action_fft_freq=1.339 joint_speed_mean=0.437 rod_speed_mean=0.029 contact_switch_hz=1.722 foot_slip=0.082 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=21.946 slip_per_m=1.064 work_per_m=1061.041 cam_work_per_m=152.288
       d090: action_jitter=0.113 action_fft_freq=1.325 joint_speed_mean=0.384 rod_speed_mean=0.029 contact_switch_hz=1.694 foot_slip=0.075 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=21.770 slip_per_m=1.004 work_per_m=1034.702 cam_work_per_m=145.111
       d112: action_jitter=0.118 action_fft_freq=1.337 joint_speed_mean=0.444 rod_speed_mean=0.029 contact_switch_hz=1.861 foot_slip=0.079 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=23.843 slip_per_m=1.033 work_per_m=1054.936 cam_work_per_m=164.485
       d135: action_jitter=0.135 action_fft_freq=1.344 joint_speed_mean=0.553 rod_speed_mean=0.029 contact_switch_hz=1.972 foot_slip=0.083 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=25.832 slip_per_m=1.065 work_per_m=1028.854 cam_work_per_m=248.951
       d158: action_jitter=0.146 action_fft_freq=1.341 joint_speed_mean=0.654 rod_speed_mean=0.029 contact_switch_hz=1.819 foot_slip=0.082 wz_active_frac=0.006 clock_restarts=0.000 swings_per_m=25.159 slip_per_m=1.132 work_per_m=1113.648 cam_work_per_m=311.488
       d180: action_jitter=0.141 action_fft_freq=1.245 joint_speed_mean=0.705 rod_speed_mean=0.029 contact_switch_hz=1.889 foot_slip=0.094 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=26.308 slip_per_m=1.289 work_per_m=1210.493 cam_work_per_m=395.748
       d202: action_jitter=0.133 action_fft_freq=1.232 joint_speed_mean=0.668 rod_speed_mean=0.029 contact_switch_hz=1.889 foot_slip=0.092 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=26.202 slip_per_m=1.293 work_per_m=1183.977 cam_work_per_m=384.679
       d225: action_jitter=0.122 action_fft_freq=1.228 joint_speed_mean=0.584 rod_speed_mean=0.029 contact_switch_hz=1.826 foot_slip=0.082 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=24.850 slip_per_m=1.159 work_per_m=1117.379 cam_work_per_m=278.289
       d248: action_jitter=0.108 action_fft_freq=1.179 joint_speed_mean=0.458 rod_speed_mean=0.029 contact_switch_hz=1.500 foot_slip=0.074 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=19.770 slip_per_m=0.986 work_per_m=1023.051 cam_work_per_m=247.381
       d270: action_jitter=0.101 action_fft_freq=1.148 joint_speed_mean=0.395 rod_speed_mean=0.028 contact_switch_hz=1.333 foot_slip=0.072 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=17.361 slip_per_m=0.945 work_per_m=996.550 cam_work_per_m=232.601
       d292: action_jitter=0.112 action_fft_freq=1.180 joint_speed_mean=0.511 rod_speed_mean=0.029 contact_switch_hz=1.639 foot_slip=0.079 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=21.472 slip_per_m=1.042 work_per_m=1084.024 cam_work_per_m=265.954
       d315: action_jitter=0.133 action_fft_freq=1.281 joint_speed_mean=0.608 rod_speed_mean=0.029 contact_switch_hz=1.847 foot_slip=0.083 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=24.545 slip_per_m=1.137 work_per_m=1116.329 cam_work_per_m=313.777
       d338: action_jitter=0.139 action_fft_freq=1.241 joint_speed_mean=0.699 rod_speed_mean=0.029 contact_switch_hz=1.833 foot_slip=0.088 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=24.303 slip_per_m=1.172 work_per_m=1124.833 cam_work_per_m=451.641

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
       d000: steps_per_m=11.874 support_feet=2.950 taps_per_m=12.507 swing_apex=0.115 step_rate_hz=2.667 gait_hz=0.415 clock_hz=0.314 cadence_ratio=1.321 hz_speed_slope=0.382 offset_R=0.057
       d022: steps_per_m=12.705 support_feet=2.970 taps_per_m=12.552 swing_apex=0.117 step_rate_hz=2.917 gait_hz=0.452 clock_hz=0.314 cadence_ratio=1.437 hz_speed_slope=0.058 offset_R=0.073
       d045: steps_per_m=12.898 support_feet=2.938 taps_per_m=9.530 swing_apex=0.099 step_rate_hz=3.000 gait_hz=0.476 clock_hz=0.314 cadence_ratio=1.515 hz_speed_slope=-0.246 offset_R=0.106
       d068: steps_per_m=13.088 support_feet=3.004 taps_per_m=8.554 swing_apex=0.083 step_rate_hz=3.083 gait_hz=0.452 clock_hz=0.314 cadence_ratio=1.437 hz_speed_slope=0.566 offset_R=0.159
       d090: steps_per_m=12.252 support_feet=3.107 taps_per_m=9.228 swing_apex=0.082 step_rate_hz=2.833 gait_hz=0.446 clock_hz=0.314 cadence_ratio=1.418 hz_speed_slope=-0.427 offset_R=0.111
       d112: steps_per_m=13.723 support_feet=3.012 taps_per_m=10.119 swing_apex=0.079 step_rate_hz=3.167 gait_hz=0.470 clock_hz=0.314 cadence_ratio=1.495 hz_speed_slope=-0.967 offset_R=0.217
       d135: steps_per_m=14.327 support_feet=2.930 taps_per_m=11.385 swing_apex=0.087 step_rate_hz=3.250 gait_hz=0.519 clock_hz=0.314 cadence_ratio=1.651 hz_speed_slope=0.426 offset_R=0.073
       d158: steps_per_m=13.504 support_feet=3.001 taps_per_m=11.312 swing_apex=0.107 step_rate_hz=2.917 gait_hz=0.470 clock_hz=0.314 cadence_ratio=1.487 hz_speed_slope=0.530 offset_R=0.106
       d180: steps_per_m=12.883 support_feet=2.968 taps_per_m=13.305 swing_apex=0.111 step_rate_hz=2.750 gait_hz=0.415 clock_hz=0.314 cadence_ratio=1.321 hz_speed_slope=0.651 offset_R=0.088
       d202: steps_per_m=13.532 support_feet=3.017 taps_per_m=12.571 swing_apex=0.120 step_rate_hz=2.917 gait_hz=0.415 clock_hz=0.314 cadence_ratio=1.312 hz_speed_slope=0.190 offset_R=0.203
       d225: steps_per_m=13.075 support_feet=3.082 taps_per_m=11.590 swing_apex=0.122 step_rate_hz=2.833 gait_hz=0.439 clock_hz=0.314 cadence_ratio=1.398 hz_speed_slope=-0.112 offset_R=0.061
       d248: steps_per_m=11.087 support_feet=3.037 taps_per_m=8.628 swing_apex=0.113 step_rate_hz=2.500 gait_hz=0.403 clock_hz=0.314 cadence_ratio=1.282 hz_speed_slope=2.325 offset_R=0.044
       d270: steps_per_m=10.043 support_feet=3.032 taps_per_m=7.229 swing_apex=0.115 step_rate_hz=2.333 gait_hz=0.378 clock_hz=0.314 cadence_ratio=1.204 hz_speed_slope=0.070 offset_R=0.039
       d292: steps_per_m=12.064 support_feet=3.009 taps_per_m=9.298 swing_apex=0.108 step_rate_hz=2.750 gait_hz=0.421 clock_hz=0.314 cadence_ratio=1.340 hz_speed_slope=1.871 offset_R=0.053
       d315: steps_per_m=13.319 support_feet=3.065 taps_per_m=11.127 swing_apex=0.118 step_rate_hz=3.000 gait_hz=0.446 clock_hz=0.314 cadence_ratio=1.418 hz_speed_slope=0.389 offset_R=0.067
       d338: steps_per_m=12.496 support_feet=3.001 taps_per_m=11.611 swing_apex=0.114 step_rate_hz=2.833 gait_hz=0.421 clock_hz=0.314 cadence_ratio=1.340 hz_speed_slope=-0.222 offset_R=0.063

body yaw by category (own hold, window [4 s, hold_s); medians pooled over both hands; per-hand values are in aggregate.suite.by_category[c].by_hand):
       d000: net_yaw_rate=-0.004 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.043 heading_drift=0.006 planar_drift=n/a body_ratio=1.033 cross_body=0.025 adapter_clipped_frac=n/a
       d022: net_yaw_rate=-0.002 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.025 heading_drift=0.005 planar_drift=n/a body_ratio=1.036 cross_body=0.024 adapter_clipped_frac=n/a
       d045: net_yaw_rate=-0.002 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.026 heading_drift=0.004 planar_drift=n/a body_ratio=1.065 cross_body=0.036 adapter_clipped_frac=n/a
       d068: net_yaw_rate=-0.007 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.027 heading_drift=0.007 planar_drift=n/a body_ratio=1.077 cross_body=0.033 adapter_clipped_frac=n/a
       d090: net_yaw_rate=-0.000 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.015 heading_drift=0.002 planar_drift=n/a body_ratio=1.069 cross_body=0.025 adapter_clipped_frac=n/a
       d112: net_yaw_rate=0.005 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.029 heading_drift=0.006 planar_drift=n/a body_ratio=1.070 cross_body=0.034 adapter_clipped_frac=n/a
       d135: net_yaw_rate=0.001 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.023 heading_drift=0.004 planar_drift=n/a body_ratio=1.047 cross_body=0.032 adapter_clipped_frac=n/a
       d158: net_yaw_rate=0.011 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.022 heading_drift=0.012 planar_drift=n/a body_ratio=0.984 cross_body=0.024 adapter_clipped_frac=n/a
       d180: net_yaw_rate=0.008 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.030 heading_drift=0.008 planar_drift=n/a body_ratio=0.985 cross_body=0.026 adapter_clipped_frac=n/a
       d202: net_yaw_rate=0.001 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.045 heading_drift=0.003 planar_drift=n/a body_ratio=0.997 cross_body=0.031 adapter_clipped_frac=n/a
       d225: net_yaw_rate=0.003 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.038 heading_drift=0.004 planar_drift=n/a body_ratio=1.013 cross_body=0.032 adapter_clipped_frac=n/a
       d248: net_yaw_rate=-0.001 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.025 heading_drift=0.003 planar_drift=n/a body_ratio=1.048 cross_body=0.030 adapter_clipped_frac=n/a
       d270: net_yaw_rate=0.003 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.021 heading_drift=0.004 planar_drift=n/a body_ratio=1.060 cross_body=0.027 adapter_clipped_frac=n/a
       d292: net_yaw_rate=0.006 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.041 heading_drift=0.006 planar_drift=n/a body_ratio=1.046 cross_body=0.030 adapter_clipped_frac=n/a
       d315: net_yaw_rate=0.001 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.036 heading_drift=0.004 planar_drift=n/a body_ratio=1.044 cross_body=0.028 adapter_clipped_frac=n/a
       d338: net_yaw_rate=-0.003 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.038 heading_drift=0.004 planar_drift=n/a body_ratio=1.036 cross_body=0.030 adapter_clipped_frac=n/a

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
