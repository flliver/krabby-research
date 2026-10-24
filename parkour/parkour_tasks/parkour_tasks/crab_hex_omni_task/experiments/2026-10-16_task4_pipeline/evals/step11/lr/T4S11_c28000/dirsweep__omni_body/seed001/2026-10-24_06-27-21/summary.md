```
=== crab-hex gait eval ===
scenario   : dirsweep__omni_body  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_28000.pt
episodes   : 1600  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6928555019079028  p25=0.6605337087699843  p75=0.7333823078777177
tippy_tap_fraction median=0.40789473684210525
slip_ratio         median=0.2079136515814896
tracking_ratio     median=1.0250407462836657  (achieved/commanded vx, walking holds; n=1600)
schedule_completion_rate=1.0
terminations={'schedule_complete': 1600}
facing_error_rms   median=0.03972498911980471  (rad, whole episode; n=1600)

by hold:
    settle: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.6665085142048397 (n=1530)
      d000: cmd 0.22 -> achieved 0.222 m/s | tripod median=0.6778554646601859 (n=100)
      d022: cmd 0.22 -> achieved 0.222 m/s | tripod median=0.7522551938793598 (n=100)
      d045: cmd 0.22 -> achieved 0.231 m/s | tripod median=0.7172858921941769 (n=100)
      d068: cmd 0.22 -> achieved 0.230 m/s | tripod median=0.6675799844145043 (n=100)
      d090: cmd 0.22 -> achieved 0.230 m/s | tripod median=0.667806896123577 (n=100)
      d112: cmd 0.22 -> achieved 0.230 m/s | tripod median=0.6535373787674383 (n=100)
      d135: cmd 0.22 -> achieved 0.227 m/s | tripod median=0.6559073058087519 (n=100)
      d158: cmd 0.22 -> achieved 0.218 m/s | tripod median=0.7686177890302754 (n=100)
      d180: cmd 0.22 -> achieved 0.215 m/s | tripod median=0.6829676863552507 (n=100)
      d202: cmd 0.22 -> achieved 0.217 m/s | tripod median=0.6657854412231914 (n=100)
      d225: cmd 0.22 -> achieved 0.224 m/s | tripod median=0.7196063941067612 (n=100)
      d248: cmd 0.22 -> achieved 0.230 m/s | tripod median=0.753291557153555 (n=100)
      d270: cmd 0.22 -> achieved 0.231 m/s | tripod median=0.7564344325538062 (n=100)
      d292: cmd 0.22 -> achieved 0.226 m/s | tripod median=0.7428444128026834 (n=100)
      d315: cmd 0.22 -> achieved 0.226 m/s | tripod median=0.7147852332765554 (n=100)
      d338: cmd 0.22 -> achieved 0.222 m/s | tripod median=0.660174015604964 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.015 initial=0.008 final=0.017 max=0.031 closed=n/a | settle=0.000 (n=1600/1600) | progress=n/a net=0.000 yaw_rate_ratio=n/a
      d000: err rms=0.055 initial=0.017 final=0.044 max=0.129 closed=n/a | settle=0.000 (n=99/100) | progress=-0.003 net=0.003 yaw_rate_ratio=0.715
      d022: err rms=0.052 initial=0.017 final=0.049 max=0.107 closed=n/a | settle=0.000 (n=100/100) | progress=-0.001 net=-0.001 yaw_rate_ratio=0.318
      d045: err rms=0.030 initial=0.017 final=0.030 max=0.068 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.003 yaw_rate_ratio=n/a
      d068: err rms=0.040 initial=0.019 final=0.065 max=0.085 closed=n/a | settle=0.000 (n=100/100) | progress=-0.028 net=0.006 yaw_rate_ratio=0.313
      d090: err rms=0.022 initial=0.016 final=0.020 max=0.048 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.002 yaw_rate_ratio=n/a
      d112: err rms=0.043 initial=0.016 final=0.056 max=0.091 closed=n/a | settle=0.000 (n=100/100) | progress=-0.002 net=-0.001 yaw_rate_ratio=0.270
      d135: err rms=0.032 initial=0.015 final=0.020 max=0.072 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.002 yaw_rate_ratio=n/a
      d158: err rms=0.075 initial=0.016 final=0.097 max=0.129 closed=n/a | settle=0.000 (n=97/100) | progress=-0.003 net=-0.004 yaw_rate_ratio=0.334
      d180: err rms=0.094 initial=0.019 final=0.098 max=0.177 closed=n/a | settle=11.140 (n=99/100) | progress=-0.001 net=-0.004 yaw_rate_ratio=0.574
      d202: err rms=0.058 initial=0.017 final=0.049 max=0.123 closed=n/a | settle=0.000 (n=100/100) | progress=-0.000 net=-0.000 yaw_rate_ratio=0.486
      d225: err rms=0.033 initial=0.016 final=0.024 max=0.081 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.002 yaw_rate_ratio=n/a
      d248: err rms=0.034 initial=0.017 final=0.030 max=0.074 closed=n/a | settle=0.000 (n=100/100) | progress=-0.004 net=0.001 yaw_rate_ratio=0.327
      d270: err rms=0.033 initial=0.016 final=0.042 max=0.076 closed=n/a | settle=0.000 (n=100/100) | progress=-0.003 net=-0.000 yaw_rate_ratio=0.279
      d292: err rms=0.069 initial=0.017 final=0.081 max=0.135 closed=n/a | settle=0.000 (n=99/100) | progress=-0.000 net=-0.003 yaw_rate_ratio=0.326
      d315: err rms=0.046 initial=0.015 final=0.032 max=0.094 closed=n/a | settle=0.000 (n=100/100) | progress=-0.002 net=0.000 yaw_rate_ratio=0.270
      d338: err rms=0.048 initial=0.017 final=0.045 max=0.110 closed=n/a | settle=0.000 (n=100/100) | progress=0.004 net=0.003 yaw_rate_ratio=0.525

body yaw by hold (window [4 s, hold_s); medians; net = net yaw rate rad/s, ratio = net / wz_cmd SIGNED (the gated turn number), wz_ratio = mean wz / wz_cmd; ripple = std of the 1-s mean wz; drift = |mean wz| on wz = 0 holds; planar = xy drift m/s on pure turns; body = along-bearing speed / |v_cmd|, cross = mean |perp| m/s):
          settle: net=n/a ratio=n/a wz_ratio=n/a ripple=n/a | drift=n/a planar=n/a | body=n/a cross=n/a (n=0)
            d000: net=-0.001 ratio=n/a wz_ratio=n/a ripple=0.053 | drift=0.005 planar=n/a | body=1.010 cross=0.026 (n=100)
            d022: net=0.000 ratio=n/a wz_ratio=n/a ripple=0.037 | drift=0.004 planar=n/a | body=1.006 cross=0.023 (n=100)
            d045: net=-0.005 ratio=n/a wz_ratio=n/a ripple=0.022 | drift=0.005 planar=n/a | body=1.057 cross=0.038 (n=100)
            d068: net=-0.008 ratio=n/a wz_ratio=n/a ripple=0.026 | drift=0.009 planar=n/a | body=1.049 cross=0.034 (n=100)
            d090: net=-0.002 ratio=n/a wz_ratio=n/a ripple=0.015 | drift=0.003 planar=n/a | body=1.045 cross=0.027 (n=100)
            d112: net=0.005 ratio=n/a wz_ratio=n/a ripple=0.028 | drift=0.005 planar=n/a | body=1.054 cross=0.035 (n=100)
            d135: net=-0.000 ratio=n/a wz_ratio=n/a ripple=0.025 | drift=0.004 planar=n/a | body=1.035 cross=0.034 (n=100)
            d158: net=0.005 ratio=n/a wz_ratio=n/a ripple=0.022 | drift=0.006 planar=n/a | body=0.985 cross=0.025 (n=100)
            d180: net=0.002 ratio=n/a wz_ratio=n/a ripple=0.039 | drift=0.004 planar=n/a | body=0.973 cross=0.028 (n=100)
            d202: net=-0.000 ratio=n/a wz_ratio=n/a ripple=0.047 | drift=0.003 planar=n/a | body=0.989 cross=0.032 (n=100)
            d225: net=-0.002 ratio=n/a wz_ratio=n/a ripple=0.039 | drift=0.004 planar=n/a | body=1.021 cross=0.033 (n=100)
            d248: net=0.001 ratio=n/a wz_ratio=n/a ripple=0.023 | drift=0.004 planar=n/a | body=1.053 cross=0.029 (n=100)
            d270: net=0.004 ratio=n/a wz_ratio=n/a ripple=0.021 | drift=0.004 planar=n/a | body=1.049 cross=0.027 (n=100)
            d292: net=0.005 ratio=n/a wz_ratio=n/a ripple=0.042 | drift=0.006 planar=n/a | body=1.031 cross=0.031 (n=100)
            d315: net=-0.001 ratio=n/a wz_ratio=n/a ripple=0.042 | drift=0.003 planar=n/a | body=1.037 cross=0.028 (n=100)
            d338: net=-0.001 ratio=n/a wz_ratio=n/a ripple=0.042 | drift=0.004 planar=n/a | body=1.007 cross=0.028 (n=100)

by category (16 waves, 1600 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: n/a):
       d000: n=100 surv=1.000 (d000=1.00) | ratio=1.011 achieved=0.222 tripod=0.678 | closed=n/a progress=-0.003 net=0.003 | drift=0.223 dir_err=0.028
       d022: n=100 surv=1.000 (d022=1.00) | ratio=1.008 achieved=0.222 tripod=0.752 | closed=n/a progress=-0.001 net=-0.001 | drift=0.224 dir_err=0.012
       d045: n=100 surv=1.000 (d045=1.00) | ratio=1.050 achieved=0.231 tripod=0.717 | closed=n/a progress=n/a net=0.003 | drift=0.236 dir_err=0.072
       d068: n=100 surv=1.000 (d068=1.00) | ratio=1.046 achieved=0.230 tripod=0.668 | closed=n/a progress=-0.028 net=0.006 | drift=0.235 dir_err=0.034
       d090: n=100 surv=1.000 (d090=1.00) | ratio=1.045 achieved=0.230 tripod=0.668 | closed=n/a progress=n/a net=0.002 | drift=0.233 dir_err=0.011
       d112: n=100 surv=1.000 (d112=1.00) | ratio=1.047 achieved=0.230 tripod=0.654 | closed=n/a progress=-0.002 net=-0.001 | drift=0.235 dir_err=0.031
       d135: n=100 surv=1.000 (d135=1.00) | ratio=1.033 achieved=0.227 tripod=0.656 | closed=n/a progress=n/a net=0.002 | drift=0.231 dir_err=0.055
       d158: n=100 surv=1.000 (d158=1.00) | ratio=0.990 achieved=0.218 tripod=0.769 | closed=n/a progress=-0.003 net=-0.004 | drift=0.220 dir_err=0.017
       d180: n=100 surv=1.000 (d180=1.00) | ratio=0.979 achieved=0.215 tripod=0.683 | closed=n/a progress=-0.001 net=-0.004 | drift=0.216 dir_err=0.036
       d202: n=100 surv=1.000 (d202=1.00) | ratio=0.985 achieved=0.217 tripod=0.666 | closed=n/a progress=-0.000 net=-0.000 | drift=0.219 dir_err=0.019
       d225: n=100 surv=1.000 (d225=1.00) | ratio=1.017 achieved=0.224 tripod=0.720 | closed=n/a progress=n/a net=0.002 | drift=0.226 dir_err=0.036
       d248: n=100 surv=1.000 (d248=1.00) | ratio=1.043 achieved=0.230 tripod=0.753 | closed=n/a progress=-0.004 net=0.001 | drift=0.231 dir_err=0.051
       d270: n=100 surv=1.000 (d270=1.00) | ratio=1.048 achieved=0.231 tripod=0.756 | closed=n/a progress=-0.003 net=-0.000 | drift=0.232 dir_err=0.027
       d292: n=100 surv=1.000 (d292=1.00) | ratio=1.026 achieved=0.226 tripod=0.743 | closed=n/a progress=-0.000 net=-0.003 | drift=0.228 dir_err=0.010
       d315: n=100 surv=1.000 (d315=1.00) | ratio=1.028 achieved=0.226 tripod=0.715 | closed=n/a progress=-0.002 net=0.000 | drift=0.227 dir_err=0.012
       d338: n=100 surv=1.000 (d338=1.00) | ratio=1.007 achieved=0.222 tripod=0.660 | closed=n/a progress=0.004 net=0.003 | drift=0.222 dir_err=0.011

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
       d000: action_jitter=0.110 action_fft_freq=1.112 joint_speed_mean=0.720 rod_speed_mean=0.029 contact_switch_hz=1.799 foot_slip=0.090 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=24.818 slip_per_m=1.202 work_per_m=1194.149 cam_work_per_m=437.004
       d022: action_jitter=0.126 action_fft_freq=1.270 joint_speed_mean=0.669 rod_speed_mean=0.029 contact_switch_hz=1.875 foot_slip=0.083 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=25.399 slip_per_m=1.083 work_per_m=1146.534 cam_work_per_m=288.030
       d045: action_jitter=0.106 action_fft_freq=1.259 joint_speed_mean=0.555 rod_speed_mean=0.029 contact_switch_hz=1.660 foot_slip=0.081 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=21.457 slip_per_m=1.028 work_per_m=1089.810 cam_work_per_m=215.820
       d068: action_jitter=0.091 action_fft_freq=1.242 joint_speed_mean=0.434 rod_speed_mean=0.029 contact_switch_hz=1.569 foot_slip=0.081 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=20.732 slip_per_m=1.059 work_per_m=1094.101 cam_work_per_m=150.449
       d090: action_jitter=0.087 action_fft_freq=1.222 joint_speed_mean=0.378 rod_speed_mean=0.029 contact_switch_hz=1.556 foot_slip=0.076 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=20.545 slip_per_m=1.002 work_per_m=1053.688 cam_work_per_m=144.232
       d112: action_jitter=0.092 action_fft_freq=1.260 joint_speed_mean=0.447 rod_speed_mean=0.029 contact_switch_hz=1.771 foot_slip=0.079 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=23.176 slip_per_m=1.047 work_per_m=1071.843 cam_work_per_m=168.407
       d135: action_jitter=0.107 action_fft_freq=1.232 joint_speed_mean=0.555 rod_speed_mean=0.029 contact_switch_hz=1.861 foot_slip=0.084 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=24.663 slip_per_m=1.085 work_per_m=1049.708 cam_work_per_m=237.938
       d158: action_jitter=0.123 action_fft_freq=1.278 joint_speed_mean=0.653 rod_speed_mean=0.029 contact_switch_hz=1.771 foot_slip=0.079 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=24.165 slip_per_m=1.090 work_per_m=1116.897 cam_work_per_m=289.010
       d180: action_jitter=0.111 action_fft_freq=1.135 joint_speed_mean=0.704 rod_speed_mean=0.029 contact_switch_hz=1.799 foot_slip=0.094 wz_active_frac=0.105 clock_restarts=0.000 swings_per_m=25.381 slip_per_m=1.311 work_per_m=1230.930 cam_work_per_m=415.963
       d202: action_jitter=0.099 action_fft_freq=1.104 joint_speed_mean=0.674 rod_speed_mean=0.029 contact_switch_hz=1.778 foot_slip=0.093 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=24.883 slip_per_m=1.320 work_per_m=1217.594 cam_work_per_m=402.354
       d225: action_jitter=0.091 action_fft_freq=1.100 joint_speed_mean=0.592 rod_speed_mean=0.029 contact_switch_hz=1.722 foot_slip=0.085 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=23.267 slip_per_m=1.179 work_per_m=1142.703 cam_work_per_m=284.227
       d248: action_jitter=0.081 action_fft_freq=1.065 joint_speed_mean=0.460 rod_speed_mean=0.029 contact_switch_hz=1.444 foot_slip=0.079 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=19.080 slip_per_m=1.045 work_per_m=1048.152 cam_work_per_m=253.368
       d270: action_jitter=0.077 action_fft_freq=1.024 joint_speed_mean=0.390 rod_speed_mean=0.028 contact_switch_hz=1.278 foot_slip=0.074 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=16.880 slip_per_m=0.992 work_per_m=1038.925 cam_work_per_m=239.146
       d292: action_jitter=0.083 action_fft_freq=1.075 joint_speed_mean=0.503 rod_speed_mean=0.029 contact_switch_hz=1.611 foot_slip=0.082 wz_active_frac=0.008 clock_restarts=0.000 swings_per_m=21.418 slip_per_m=1.096 work_per_m=1127.872 cam_work_per_m=267.936
       d315: action_jitter=0.095 action_fft_freq=1.156 joint_speed_mean=0.608 rod_speed_mean=0.029 contact_switch_hz=1.861 foot_slip=0.085 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=24.825 slip_per_m=1.175 work_per_m=1142.232 cam_work_per_m=322.400
       d338: action_jitter=0.101 action_fft_freq=1.117 joint_speed_mean=0.695 rod_speed_mean=0.029 contact_switch_hz=1.819 foot_slip=0.091 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=25.020 slip_per_m=1.232 work_per_m=1167.851 cam_work_per_m=486.452

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
       d000: steps_per_m=12.416 support_feet=2.976 taps_per_m=11.955 swing_apex=0.115 step_rate_hz=2.750 gait_hz=0.415 clock_hz=0.314 cadence_ratio=1.321 hz_speed_slope=0.625 offset_R=0.076
       d022: steps_per_m=12.883 support_feet=2.954 taps_per_m=12.180 swing_apex=0.122 step_rate_hz=2.833 gait_hz=0.452 clock_hz=0.314 cadence_ratio=1.435 hz_speed_slope=-0.599 offset_R=0.097
       d045: steps_per_m=13.102 support_feet=2.942 taps_per_m=8.167 swing_apex=0.099 step_rate_hz=3.000 gait_hz=0.476 clock_hz=0.314 cadence_ratio=1.515 hz_speed_slope=-0.079 offset_R=0.062
       d068: steps_per_m=13.242 support_feet=2.983 taps_per_m=7.002 swing_apex=0.088 step_rate_hz=3.083 gait_hz=0.446 clock_hz=0.314 cadence_ratio=1.418 hz_speed_slope=0.989 offset_R=0.062
       d090: steps_per_m=12.641 support_feet=3.088 taps_per_m=8.176 swing_apex=0.085 step_rate_hz=2.917 gait_hz=0.439 clock_hz=0.314 cadence_ratio=1.398 hz_speed_slope=-1.153 offset_R=0.078
       d112: steps_per_m=13.750 support_feet=3.015 taps_per_m=9.380 swing_apex=0.081 step_rate_hz=3.167 gait_hz=0.464 clock_hz=0.314 cadence_ratio=1.476 hz_speed_slope=-0.663 offset_R=0.145
       d135: steps_per_m=14.458 support_feet=2.923 taps_per_m=9.834 swing_apex=0.090 step_rate_hz=3.292 gait_hz=0.513 clock_hz=0.314 cadence_ratio=1.631 hz_speed_slope=-0.282 offset_R=0.120
       d158: steps_per_m=13.180 support_feet=2.994 taps_per_m=11.111 swing_apex=0.113 step_rate_hz=2.833 gait_hz=0.464 clock_hz=0.314 cadence_ratio=1.472 hz_speed_slope=0.429 offset_R=0.128
       d180: steps_per_m=12.905 support_feet=2.967 taps_per_m=12.119 swing_apex=0.112 step_rate_hz=2.750 gait_hz=0.409 clock_hz=0.322 cadence_ratio=1.267 hz_speed_slope=0.721 offset_R=0.058
       d202: steps_per_m=13.596 support_feet=3.003 taps_per_m=11.309 swing_apex=0.120 step_rate_hz=2.917 gait_hz=0.403 clock_hz=0.314 cadence_ratio=1.282 hz_speed_slope=0.029 offset_R=0.041
       d225: steps_per_m=13.202 support_feet=3.073 taps_per_m=10.002 swing_apex=0.125 step_rate_hz=2.917 gait_hz=0.433 clock_hz=0.314 cadence_ratio=1.379 hz_speed_slope=-0.076 offset_R=0.012
       d248: steps_per_m=11.056 support_feet=3.032 taps_per_m=8.063 swing_apex=0.116 step_rate_hz=2.500 gait_hz=0.391 clock_hz=0.314 cadence_ratio=1.243 hz_speed_slope=1.468 offset_R=0.072
       d270: steps_per_m=9.656 support_feet=3.023 taps_per_m=7.017 swing_apex=0.116 step_rate_hz=2.167 gait_hz=0.348 clock_hz=0.314 cadence_ratio=1.107 hz_speed_slope=2.501 offset_R=0.064
       d292: steps_per_m=12.052 support_feet=2.993 taps_per_m=9.146 swing_apex=0.111 step_rate_hz=2.667 gait_hz=0.403 clock_hz=0.315 cadence_ratio=1.276 hz_speed_slope=1.572 offset_R=0.156
       d315: steps_per_m=13.717 support_feet=3.079 taps_per_m=10.680 swing_apex=0.116 step_rate_hz=3.083 gait_hz=0.452 clock_hz=0.314 cadence_ratio=1.437 hz_speed_slope=-0.426 offset_R=0.023
       d338: steps_per_m=13.049 support_feet=3.027 taps_per_m=11.638 swing_apex=0.115 step_rate_hz=2.833 gait_hz=0.427 clock_hz=0.314 cadence_ratio=1.359 hz_speed_slope=-0.316 offset_R=0.107

body yaw by category (own hold, window [4 s, hold_s); medians pooled over both hands; per-hand values are in aggregate.suite.by_category[c].by_hand):
       d000: net_yaw_rate=-0.001 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.053 heading_drift=0.005 planar_drift=n/a body_ratio=1.010 cross_body=0.026 adapter_clipped_frac=n/a
       d022: net_yaw_rate=0.000 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.037 heading_drift=0.004 planar_drift=n/a body_ratio=1.006 cross_body=0.023 adapter_clipped_frac=n/a
       d045: net_yaw_rate=-0.005 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.022 heading_drift=0.005 planar_drift=n/a body_ratio=1.057 cross_body=0.038 adapter_clipped_frac=n/a
       d068: net_yaw_rate=-0.008 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.026 heading_drift=0.009 planar_drift=n/a body_ratio=1.049 cross_body=0.034 adapter_clipped_frac=n/a
       d090: net_yaw_rate=-0.002 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.015 heading_drift=0.003 planar_drift=n/a body_ratio=1.045 cross_body=0.027 adapter_clipped_frac=n/a
       d112: net_yaw_rate=0.005 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.028 heading_drift=0.005 planar_drift=n/a body_ratio=1.054 cross_body=0.035 adapter_clipped_frac=n/a
       d135: net_yaw_rate=-0.000 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.025 heading_drift=0.004 planar_drift=n/a body_ratio=1.035 cross_body=0.034 adapter_clipped_frac=n/a
       d158: net_yaw_rate=0.005 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.022 heading_drift=0.006 planar_drift=n/a body_ratio=0.985 cross_body=0.025 adapter_clipped_frac=n/a
       d180: net_yaw_rate=0.002 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.039 heading_drift=0.004 planar_drift=n/a body_ratio=0.973 cross_body=0.028 adapter_clipped_frac=n/a
       d202: net_yaw_rate=-0.000 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.047 heading_drift=0.003 planar_drift=n/a body_ratio=0.989 cross_body=0.032 adapter_clipped_frac=n/a
       d225: net_yaw_rate=-0.002 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.039 heading_drift=0.004 planar_drift=n/a body_ratio=1.021 cross_body=0.033 adapter_clipped_frac=n/a
       d248: net_yaw_rate=0.001 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.023 heading_drift=0.004 planar_drift=n/a body_ratio=1.053 cross_body=0.029 adapter_clipped_frac=n/a
       d270: net_yaw_rate=0.004 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.021 heading_drift=0.004 planar_drift=n/a body_ratio=1.049 cross_body=0.027 adapter_clipped_frac=n/a
       d292: net_yaw_rate=0.005 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.042 heading_drift=0.006 planar_drift=n/a body_ratio=1.031 cross_body=0.031 adapter_clipped_frac=n/a
       d315: net_yaw_rate=-0.001 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.042 heading_drift=0.003 planar_drift=n/a body_ratio=1.037 cross_body=0.028 adapter_clipped_frac=n/a
       d338: net_yaw_rate=-0.001 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.042 heading_drift=0.004 planar_drift=n/a body_ratio=1.007 cross_body=0.028 adapter_clipped_frac=n/a

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
