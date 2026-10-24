```
=== crab-hex gait eval ===
scenario   : dirsweep__omni_body  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt
episodes   : 1600  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.709947002406682  p25=0.6775705457296003  p75=0.7438949097292802
tippy_tap_fraction median=0.44047619047619047
slip_ratio         median=0.23279867311228802
tracking_ratio     median=1.0388545229153279  (achieved/commanded vx, walking holds; n=1600)
schedule_completion_rate=1.0
terminations={'schedule_complete': 1600}
facing_error_rms   median=0.03625078116211186  (rad, whole episode; n=1600)

by hold:
    settle: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.767403238365936 (n=1557)
      d000: cmd 0.22 -> achieved 0.228 m/s | tripod median=0.6996679582586858 (n=100)
      d022: cmd 0.22 -> achieved 0.228 m/s | tripod median=0.7548360683494675 (n=100)
      d045: cmd 0.22 -> achieved 0.233 m/s | tripod median=0.7023125264949386 (n=100)
      d068: cmd 0.22 -> achieved 0.236 m/s | tripod median=0.6662568931814825 (n=100)
      d090: cmd 0.22 -> achieved 0.235 m/s | tripod median=0.6805998971489122 (n=100)
      d112: cmd 0.22 -> achieved 0.234 m/s | tripod median=0.664769888095168 (n=100)
      d135: cmd 0.22 -> achieved 0.229 m/s | tripod median=0.6402559450582443 (n=100)
      d158: cmd 0.22 -> achieved 0.218 m/s | tripod median=0.7558459558337405 (n=100)
      d180: cmd 0.22 -> achieved 0.218 m/s | tripod median=0.7019675810920994 (n=100)
      d202: cmd 0.22 -> achieved 0.217 m/s | tripod median=0.6738594649164289 (n=100)
      d225: cmd 0.22 -> achieved 0.222 m/s | tripod median=0.7133429080544273 (n=100)
      d248: cmd 0.22 -> achieved 0.229 m/s | tripod median=0.7520977680532759 (n=100)
      d270: cmd 0.22 -> achieved 0.232 m/s | tripod median=0.7615425555687294 (n=100)
      d292: cmd 0.22 -> achieved 0.229 m/s | tripod median=0.7388848409711593 (n=100)
      d315: cmd 0.22 -> achieved 0.228 m/s | tripod median=0.7342186627658629 (n=100)
      d338: cmd 0.22 -> achieved 0.227 m/s | tripod median=0.6658237192964309 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.014 initial=0.008 final=0.012 max=0.028 closed=n/a | settle=0.000 (n=1600/1600) | progress=n/a net=0.001 yaw_rate_ratio=n/a
      d000: err rms=0.046 initial=0.013 final=0.043 max=0.105 closed=n/a | settle=0.000 (n=99/100) | progress=-0.000 net=0.002 yaw_rate_ratio=0.420
      d022: err rms=0.051 initial=0.014 final=0.036 max=0.099 closed=n/a | settle=0.000 (n=100/100) | progress=-0.000 net=0.001 yaw_rate_ratio=0.410
      d045: err rms=0.028 initial=0.010 final=0.022 max=0.065 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a
      d068: err rms=0.040 initial=0.012 final=0.058 max=0.085 closed=n/a | settle=0.000 (n=100/100) | progress=-0.032 net=0.002 yaw_rate_ratio=0.166
      d090: err rms=0.018 initial=0.013 final=0.015 max=0.042 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
      d112: err rms=0.046 initial=0.014 final=0.053 max=0.093 closed=n/a | settle=0.000 (n=100/100) | progress=-0.006 net=-0.001 yaw_rate_ratio=0.227
      d135: err rms=0.029 initial=0.012 final=0.023 max=0.070 closed=n/a | settle=0.000 (n=100/100) | progress=-0.004 net=0.001 yaw_rate_ratio=0.467
      d158: err rms=0.071 initial=0.014 final=0.103 max=0.136 closed=n/a | settle=0.000 (n=100/100) | progress=-0.003 net=-0.001 yaw_rate_ratio=0.345
      d180: err rms=0.056 initial=0.014 final=0.062 max=0.120 closed=n/a | settle=0.000 (n=100/100) | progress=-0.002 net=0.003 yaw_rate_ratio=0.504
      d202: err rms=0.041 initial=0.011 final=0.026 max=0.099 closed=n/a | settle=0.000 (n=100/100) | progress=-0.003 net=0.001 yaw_rate_ratio=0.409
      d225: err rms=0.030 initial=0.012 final=0.018 max=0.070 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
      d248: err rms=0.035 initial=0.012 final=0.040 max=0.076 closed=n/a | settle=0.000 (n=100/100) | progress=0.009 net=0.001 yaw_rate_ratio=0.719
      d270: err rms=0.029 initial=0.014 final=0.030 max=0.071 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a
      d292: err rms=0.064 initial=0.011 final=0.091 max=0.125 closed=n/a | settle=0.000 (n=99/100) | progress=-0.002 net=-0.003 yaw_rate_ratio=0.339
      d315: err rms=0.049 initial=0.013 final=0.043 max=0.101 closed=n/a | settle=0.000 (n=100/100) | progress=-0.001 net=0.001 yaw_rate_ratio=0.294
      d338: err rms=0.035 initial=0.011 final=0.025 max=0.090 closed=n/a | settle=0.000 (n=99/100) | progress=-0.004 net=0.000 yaw_rate_ratio=0.508

body yaw by hold (window [4 s, hold_s); medians; net = net yaw rate rad/s, ratio = net / wz_cmd SIGNED (the gated turn number), wz_ratio = mean wz / wz_cmd; ripple = std of the 1-s mean wz; drift = |mean wz| on wz = 0 holds; planar = xy drift m/s on pure turns; body = along-bearing speed / |v_cmd|, cross = mean |perp| m/s):
          settle: net=n/a ratio=n/a wz_ratio=n/a ripple=n/a | drift=n/a planar=n/a | body=n/a cross=n/a (n=0)
            d000: net=-0.003 ratio=n/a wz_ratio=n/a ripple=0.045 | drift=0.006 planar=n/a | body=1.036 cross=0.024 (n=100)
            d022: net=-0.002 ratio=n/a wz_ratio=n/a ripple=0.024 | drift=0.004 planar=n/a | body=1.037 cross=0.023 (n=100)
            d045: net=-0.003 ratio=n/a wz_ratio=n/a ripple=0.025 | drift=0.004 planar=n/a | body=1.066 cross=0.037 (n=100)
            d068: net=-0.008 ratio=n/a wz_ratio=n/a ripple=0.028 | drift=0.008 planar=n/a | body=1.073 cross=0.033 (n=100)
            d090: net=-0.000 ratio=n/a wz_ratio=n/a ripple=0.016 | drift=0.002 planar=n/a | body=1.070 cross=0.025 (n=100)
            d112: net=0.005 ratio=n/a wz_ratio=n/a ripple=0.029 | drift=0.005 planar=n/a | body=1.071 cross=0.033 (n=100)
            d135: net=0.002 ratio=n/a wz_ratio=n/a ripple=0.025 | drift=0.003 planar=n/a | body=1.041 cross=0.031 (n=100)
            d158: net=0.008 ratio=n/a wz_ratio=n/a ripple=0.020 | drift=0.009 planar=n/a | body=0.987 cross=0.024 (n=100)
            d180: net=0.007 ratio=n/a wz_ratio=n/a ripple=0.033 | drift=0.007 planar=n/a | body=0.985 cross=0.027 (n=100)
            d202: net=0.001 ratio=n/a wz_ratio=n/a ripple=0.046 | drift=0.003 planar=n/a | body=0.988 cross=0.031 (n=100)
            d225: net=0.003 ratio=n/a wz_ratio=n/a ripple=0.036 | drift=0.004 planar=n/a | body=1.012 cross=0.032 (n=100)
            d248: net=-0.001 ratio=n/a wz_ratio=n/a ripple=0.025 | drift=0.003 planar=n/a | body=1.048 cross=0.029 (n=100)
            d270: net=0.002 ratio=n/a wz_ratio=n/a ripple=0.019 | drift=0.003 planar=n/a | body=1.061 cross=0.026 (n=100)
            d292: net=0.006 ratio=n/a wz_ratio=n/a ripple=0.039 | drift=0.007 planar=n/a | body=1.044 cross=0.029 (n=100)
            d315: net=0.000 ratio=n/a wz_ratio=n/a ripple=0.037 | drift=0.003 planar=n/a | body=1.041 cross=0.028 (n=100)
            d338: net=-0.001 ratio=n/a wz_ratio=n/a ripple=0.042 | drift=0.004 planar=n/a | body=1.039 cross=0.029 (n=100)

by category (16 waves, 1600 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: n/a):
       d000: n=100 surv=1.000 (d000=1.00) | ratio=1.035 achieved=0.228 tripod=0.700 | closed=n/a progress=-0.000 net=0.002 | drift=0.228 dir_err=0.010
       d022: n=100 surv=1.000 (d022=1.00) | ratio=1.035 achieved=0.228 tripod=0.755 | closed=n/a progress=-0.000 net=0.001 | drift=0.229 dir_err=0.021
       d045: n=100 surv=1.000 (d045=1.00) | ratio=1.061 achieved=0.233 tripod=0.702 | closed=n/a progress=n/a net=0.000 | drift=0.237 dir_err=0.076
       d068: n=100 surv=1.000 (d068=1.00) | ratio=1.070 achieved=0.236 tripod=0.666 | closed=n/a progress=-0.032 net=0.002 | drift=0.238 dir_err=0.043
       d090: n=100 surv=1.000 (d090=1.00) | ratio=1.068 achieved=0.235 tripod=0.681 | closed=n/a progress=n/a net=0.001 | drift=0.236 dir_err=0.009
       d112: n=100 surv=1.000 (d112=1.00) | ratio=1.065 achieved=0.234 tripod=0.665 | closed=n/a progress=-0.006 net=-0.001 | drift=0.236 dir_err=0.045
       d135: n=100 surv=1.000 (d135=1.00) | ratio=1.042 achieved=0.229 tripod=0.640 | closed=n/a progress=-0.004 net=0.001 | drift=0.231 dir_err=0.066
       d158: n=100 surv=1.000 (d158=1.00) | ratio=0.993 achieved=0.218 tripod=0.756 | closed=n/a progress=-0.003 net=-0.001 | drift=0.220 dir_err=0.027
       d180: n=100 surv=1.000 (d180=1.00) | ratio=0.989 achieved=0.218 tripod=0.702 | closed=n/a progress=-0.002 net=0.003 | drift=0.219 dir_err=0.022
       d202: n=100 surv=1.000 (d202=1.00) | ratio=0.985 achieved=0.217 tripod=0.674 | closed=n/a progress=-0.003 net=0.001 | drift=0.218 dir_err=0.022
       d225: n=100 surv=1.000 (d225=1.00) | ratio=1.011 achieved=0.222 tripod=0.713 | closed=n/a progress=n/a net=0.001 | drift=0.224 dir_err=0.039
       d248: n=100 surv=1.000 (d248=1.00) | ratio=1.041 achieved=0.229 tripod=0.752 | closed=n/a progress=0.009 net=0.001 | drift=0.230 dir_err=0.051
       d270: n=100 surv=1.000 (d270=1.00) | ratio=1.054 achieved=0.232 tripod=0.762 | closed=n/a progress=n/a net=0.000 | drift=0.233 dir_err=0.031
       d292: n=100 surv=1.000 (d292=1.00) | ratio=1.042 achieved=0.229 tripod=0.739 | closed=n/a progress=-0.002 net=-0.003 | drift=0.230 dir_err=0.012
       d315: n=100 surv=1.000 (d315=1.00) | ratio=1.035 achieved=0.228 tripod=0.734 | closed=n/a progress=-0.001 net=0.001 | drift=0.228 dir_err=0.030
       d338: n=100 surv=1.000 (d338=1.00) | ratio=1.033 achieved=0.227 tripod=0.666 | closed=n/a progress=-0.004 net=0.000 | drift=0.227 dir_err=0.038

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
       d000: action_jitter=0.145 action_fft_freq=1.242 joint_speed_mean=0.722 rod_speed_mean=0.029 contact_switch_hz=1.882 foot_slip=0.088 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=24.780 slip_per_m=1.156 work_per_m=1147.800 cam_work_per_m=402.129
       d022: action_jitter=0.153 action_fft_freq=1.355 joint_speed_mean=0.675 rod_speed_mean=0.029 contact_switch_hz=1.882 foot_slip=0.083 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=24.897 slip_per_m=1.090 work_per_m=1106.132 cam_work_per_m=302.122
       d045: action_jitter=0.131 action_fft_freq=1.340 joint_speed_mean=0.561 rod_speed_mean=0.029 contact_switch_hz=1.778 foot_slip=0.083 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=22.739 slip_per_m=1.040 work_per_m=1058.454 cam_work_per_m=225.530
       d068: action_jitter=0.116 action_fft_freq=1.327 joint_speed_mean=0.439 rod_speed_mean=0.029 contact_switch_hz=1.694 foot_slip=0.082 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=21.678 slip_per_m=1.064 work_per_m=1063.219 cam_work_per_m=154.749
       d090: action_jitter=0.114 action_fft_freq=1.326 joint_speed_mean=0.383 rod_speed_mean=0.029 contact_switch_hz=1.701 foot_slip=0.075 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=22.045 slip_per_m=0.998 work_per_m=1040.340 cam_work_per_m=145.611
       d112: action_jitter=0.118 action_fft_freq=1.342 joint_speed_mean=0.447 rod_speed_mean=0.029 contact_switch_hz=1.861 foot_slip=0.080 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=24.040 slip_per_m=1.032 work_per_m=1049.291 cam_work_per_m=165.452
       d135: action_jitter=0.136 action_fft_freq=1.339 joint_speed_mean=0.554 rod_speed_mean=0.029 contact_switch_hz=1.958 foot_slip=0.082 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=25.892 slip_per_m=1.054 work_per_m=1034.055 cam_work_per_m=260.295
       d158: action_jitter=0.146 action_fft_freq=1.344 joint_speed_mean=0.654 rod_speed_mean=0.029 contact_switch_hz=1.792 foot_slip=0.081 wz_active_frac=0.007 clock_restarts=0.000 swings_per_m=24.875 slip_per_m=1.109 work_per_m=1120.170 cam_work_per_m=303.307
       d180: action_jitter=0.140 action_fft_freq=1.247 joint_speed_mean=0.707 rod_speed_mean=0.029 contact_switch_hz=1.917 foot_slip=0.093 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=26.674 slip_per_m=1.290 work_per_m=1207.186 cam_work_per_m=399.162
       d202: action_jitter=0.132 action_fft_freq=1.228 joint_speed_mean=0.667 rod_speed_mean=0.029 contact_switch_hz=1.861 foot_slip=0.091 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=26.152 slip_per_m=1.280 work_per_m=1182.229 cam_work_per_m=391.149
       d225: action_jitter=0.122 action_fft_freq=1.227 joint_speed_mean=0.584 rod_speed_mean=0.029 contact_switch_hz=1.847 foot_slip=0.081 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=24.946 slip_per_m=1.137 work_per_m=1120.083 cam_work_per_m=279.198
       d248: action_jitter=0.108 action_fft_freq=1.173 joint_speed_mean=0.460 rod_speed_mean=0.029 contact_switch_hz=1.507 foot_slip=0.075 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=19.916 slip_per_m=0.997 work_per_m=1025.008 cam_work_per_m=251.841
       d270: action_jitter=0.102 action_fft_freq=1.154 joint_speed_mean=0.396 rod_speed_mean=0.028 contact_switch_hz=1.333 foot_slip=0.073 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=17.445 slip_per_m=0.962 work_per_m=995.155 cam_work_per_m=231.111
       d292: action_jitter=0.113 action_fft_freq=1.183 joint_speed_mean=0.507 rod_speed_mean=0.029 contact_switch_hz=1.639 foot_slip=0.078 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=21.655 slip_per_m=1.029 work_per_m=1087.665 cam_work_per_m=263.564
       d315: action_jitter=0.133 action_fft_freq=1.280 joint_speed_mean=0.609 rod_speed_mean=0.029 contact_switch_hz=1.889 foot_slip=0.081 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=25.040 slip_per_m=1.107 work_per_m=1109.813 cam_work_per_m=315.150
       d338: action_jitter=0.138 action_fft_freq=1.235 joint_speed_mean=0.701 rod_speed_mean=0.029 contact_switch_hz=1.819 foot_slip=0.088 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=24.286 slip_per_m=1.161 work_per_m=1115.231 cam_work_per_m=444.018

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
       d000: steps_per_m=12.178 support_feet=2.953 taps_per_m=12.255 swing_apex=0.114 step_rate_hz=2.750 gait_hz=0.415 clock_hz=0.314 cadence_ratio=1.321 hz_speed_slope=0.667 offset_R=0.036
       d022: steps_per_m=12.721 support_feet=2.968 taps_per_m=12.084 swing_apex=0.116 step_rate_hz=2.917 gait_hz=0.452 clock_hz=0.314 cadence_ratio=1.437 hz_speed_slope=-0.362 offset_R=0.130
       d045: steps_per_m=12.776 support_feet=2.937 taps_per_m=9.877 swing_apex=0.100 step_rate_hz=3.000 gait_hz=0.476 clock_hz=0.314 cadence_ratio=1.515 hz_speed_slope=0.105 offset_R=0.164
       d068: steps_per_m=13.195 support_feet=3.008 taps_per_m=8.608 swing_apex=0.084 step_rate_hz=3.083 gait_hz=0.452 clock_hz=0.314 cadence_ratio=1.437 hz_speed_slope=0.869 offset_R=0.118
       d090: steps_per_m=12.212 support_feet=3.110 taps_per_m=9.365 swing_apex=0.083 step_rate_hz=2.833 gait_hz=0.446 clock_hz=0.314 cadence_ratio=1.418 hz_speed_slope=-0.266 offset_R=0.122
       d112: steps_per_m=13.763 support_feet=3.013 taps_per_m=10.037 swing_apex=0.079 step_rate_hz=3.167 gait_hz=0.470 clock_hz=0.314 cadence_ratio=1.495 hz_speed_slope=-0.990 offset_R=0.087
       d135: steps_per_m=14.364 support_feet=2.930 taps_per_m=11.193 swing_apex=0.087 step_rate_hz=3.250 gait_hz=0.513 clock_hz=0.314 cadence_ratio=1.631 hz_speed_slope=0.675 offset_R=0.089
       d158: steps_per_m=13.240 support_feet=3.000 taps_per_m=11.353 swing_apex=0.109 step_rate_hz=2.833 gait_hz=0.464 clock_hz=0.315 cadence_ratio=1.476 hz_speed_slope=0.635 offset_R=0.069
       d180: steps_per_m=12.913 support_feet=2.963 taps_per_m=13.804 swing_apex=0.110 step_rate_hz=2.750 gait_hz=0.415 clock_hz=0.314 cadence_ratio=1.321 hz_speed_slope=0.308 offset_R=0.177
       d202: steps_per_m=13.839 support_feet=3.015 taps_per_m=12.528 swing_apex=0.120 step_rate_hz=3.000 gait_hz=0.409 clock_hz=0.314 cadence_ratio=1.301 hz_speed_slope=0.394 offset_R=0.009
       d225: steps_per_m=13.209 support_feet=3.082 taps_per_m=11.642 swing_apex=0.123 step_rate_hz=2.917 gait_hz=0.439 clock_hz=0.314 cadence_ratio=1.398 hz_speed_slope=0.147 offset_R=0.126
       d248: steps_per_m=11.231 support_feet=3.035 taps_per_m=8.851 swing_apex=0.113 step_rate_hz=2.583 gait_hz=0.403 clock_hz=0.314 cadence_ratio=1.282 hz_speed_slope=2.142 offset_R=0.016
       d270: steps_per_m=10.004 support_feet=3.033 taps_per_m=7.195 swing_apex=0.116 step_rate_hz=2.333 gait_hz=0.378 clock_hz=0.314 cadence_ratio=1.204 hz_speed_slope=0.547 offset_R=0.071
       d292: steps_per_m=12.018 support_feet=3.005 taps_per_m=9.584 swing_apex=0.109 step_rate_hz=2.750 gait_hz=0.415 clock_hz=0.314 cadence_ratio=1.321 hz_speed_slope=2.525 offset_R=0.173
       d315: steps_per_m=13.238 support_feet=3.068 taps_per_m=11.619 swing_apex=0.117 step_rate_hz=3.000 gait_hz=0.446 clock_hz=0.314 cadence_ratio=1.418 hz_speed_slope=0.510 offset_R=0.053
       d338: steps_per_m=12.461 support_feet=2.999 taps_per_m=11.747 swing_apex=0.116 step_rate_hz=2.833 gait_hz=0.421 clock_hz=0.314 cadence_ratio=1.340 hz_speed_slope=0.426 offset_R=0.091

body yaw by category (own hold, window [4 s, hold_s); medians pooled over both hands; per-hand values are in aggregate.suite.by_category[c].by_hand):
       d000: net_yaw_rate=-0.003 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.045 heading_drift=0.006 planar_drift=n/a body_ratio=1.036 cross_body=0.024 adapter_clipped_frac=n/a
       d022: net_yaw_rate=-0.002 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.024 heading_drift=0.004 planar_drift=n/a body_ratio=1.037 cross_body=0.023 adapter_clipped_frac=n/a
       d045: net_yaw_rate=-0.003 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.025 heading_drift=0.004 planar_drift=n/a body_ratio=1.066 cross_body=0.037 adapter_clipped_frac=n/a
       d068: net_yaw_rate=-0.008 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.028 heading_drift=0.008 planar_drift=n/a body_ratio=1.073 cross_body=0.033 adapter_clipped_frac=n/a
       d090: net_yaw_rate=-0.000 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.016 heading_drift=0.002 planar_drift=n/a body_ratio=1.070 cross_body=0.025 adapter_clipped_frac=n/a
       d112: net_yaw_rate=0.005 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.029 heading_drift=0.005 planar_drift=n/a body_ratio=1.071 cross_body=0.033 adapter_clipped_frac=n/a
       d135: net_yaw_rate=0.002 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.025 heading_drift=0.003 planar_drift=n/a body_ratio=1.041 cross_body=0.031 adapter_clipped_frac=n/a
       d158: net_yaw_rate=0.008 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.020 heading_drift=0.009 planar_drift=n/a body_ratio=0.987 cross_body=0.024 adapter_clipped_frac=n/a
       d180: net_yaw_rate=0.007 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.033 heading_drift=0.007 planar_drift=n/a body_ratio=0.985 cross_body=0.027 adapter_clipped_frac=n/a
       d202: net_yaw_rate=0.001 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.046 heading_drift=0.003 planar_drift=n/a body_ratio=0.988 cross_body=0.031 adapter_clipped_frac=n/a
       d225: net_yaw_rate=0.003 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.036 heading_drift=0.004 planar_drift=n/a body_ratio=1.012 cross_body=0.032 adapter_clipped_frac=n/a
       d248: net_yaw_rate=-0.001 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.025 heading_drift=0.003 planar_drift=n/a body_ratio=1.048 cross_body=0.029 adapter_clipped_frac=n/a
       d270: net_yaw_rate=0.002 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.019 heading_drift=0.003 planar_drift=n/a body_ratio=1.061 cross_body=0.026 adapter_clipped_frac=n/a
       d292: net_yaw_rate=0.006 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.039 heading_drift=0.007 planar_drift=n/a body_ratio=1.044 cross_body=0.029 adapter_clipped_frac=n/a
       d315: net_yaw_rate=0.000 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.037 heading_drift=0.003 planar_drift=n/a body_ratio=1.041 cross_body=0.028 adapter_clipped_frac=n/a
       d338: net_yaw_rate=-0.001 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.042 heading_drift=0.004 planar_drift=n/a body_ratio=1.039 cross_body=0.029 adapter_clipped_frac=n/a

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
