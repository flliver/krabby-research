```
=== crab-hex gait eval ===
scenario   : dirsweep__omni_body  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_28000.pt
episodes   : 1600  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6956928560821868  p25=0.6586432745007799  p75=0.734621822795928
tippy_tap_fraction median=0.4069767441860465
slip_ratio         median=0.21105855465517293
tracking_ratio     median=1.0243650875587207  (achieved/commanded vx, walking holds; n=1600)
schedule_completion_rate=1.0
terminations={'schedule_complete': 1600}
facing_error_rms   median=0.038980271655506486  (rad, whole episode; n=1600)

by hold:
    settle: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.6742057157265139 (n=1488)
      d000: cmd 0.22 -> achieved 0.222 m/s | tripod median=0.6922454691396507 (n=100)
      d022: cmd 0.22 -> achieved 0.222 m/s | tripod median=0.7507063390973457 (n=100)
      d045: cmd 0.22 -> achieved 0.232 m/s | tripod median=0.7197954692633666 (n=100)
      d068: cmd 0.22 -> achieved 0.230 m/s | tripod median=0.6631807077085283 (n=100)
      d090: cmd 0.22 -> achieved 0.230 m/s | tripod median=0.6623625840889483 (n=100)
      d112: cmd 0.22 -> achieved 0.230 m/s | tripod median=0.6478788356367133 (n=100)
      d135: cmd 0.22 -> achieved 0.228 m/s | tripod median=0.6477588455366845 (n=100)
      d158: cmd 0.22 -> achieved 0.218 m/s | tripod median=0.7674103467864282 (n=100)
      d180: cmd 0.22 -> achieved 0.215 m/s | tripod median=0.6861084997784223 (n=100)
      d202: cmd 0.22 -> achieved 0.216 m/s | tripod median=0.6672652915553751 (n=100)
      d225: cmd 0.22 -> achieved 0.223 m/s | tripod median=0.7173732032721786 (n=100)
      d248: cmd 0.22 -> achieved 0.230 m/s | tripod median=0.7545094814512747 (n=100)
      d270: cmd 0.22 -> achieved 0.230 m/s | tripod median=0.7567445187656979 (n=100)
      d292: cmd 0.22 -> achieved 0.226 m/s | tripod median=0.7380067921455389 (n=100)
      d315: cmd 0.22 -> achieved 0.225 m/s | tripod median=0.7153497619675764 (n=100)
      d338: cmd 0.22 -> achieved 0.222 m/s | tripod median=0.6549849191731607 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.014 initial=0.008 final=0.016 max=0.029 closed=n/a | settle=0.000 (n=1600/1600) | progress=n/a net=-0.000 yaw_rate_ratio=n/a
      d000: err rms=0.052 initial=0.014 final=0.043 max=0.132 closed=n/a | settle=0.000 (n=100/100) | progress=-0.001 net=0.002 yaw_rate_ratio=0.668
      d022: err rms=0.050 initial=0.017 final=0.048 max=0.103 closed=n/a | settle=0.000 (n=100/100) | progress=-0.001 net=0.001 yaw_rate_ratio=0.413
      d045: err rms=0.028 initial=0.013 final=0.025 max=0.067 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.002 yaw_rate_ratio=n/a
      d068: err rms=0.042 initial=0.014 final=0.066 max=0.090 closed=n/a | settle=0.000 (n=100/100) | progress=-0.000 net=0.004 yaw_rate_ratio=0.390
      d090: err rms=0.019 initial=0.014 final=0.016 max=0.049 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.002 yaw_rate_ratio=n/a
      d112: err rms=0.042 initial=0.016 final=0.048 max=0.092 closed=n/a | settle=0.000 (n=100/100) | progress=-0.003 net=-0.001 yaw_rate_ratio=0.364
      d135: err rms=0.029 initial=0.016 final=0.025 max=0.069 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
      d158: err rms=0.074 initial=0.018 final=0.100 max=0.131 closed=n/a | settle=0.000 (n=96/100) | progress=-0.004 net=-0.003 yaw_rate_ratio=0.362
      d180: err rms=0.091 initial=0.016 final=0.102 max=0.173 closed=n/a | settle=11.160 (n=98/100) | progress=-0.001 net=-0.004 yaw_rate_ratio=0.564
      d202: err rms=0.052 initial=0.019 final=0.039 max=0.117 closed=n/a | settle=0.000 (n=100/100) | progress=-0.001 net=0.000 yaw_rate_ratio=0.545
      d225: err rms=0.034 initial=0.019 final=0.022 max=0.082 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.002 yaw_rate_ratio=n/a
      d248: err rms=0.032 initial=0.015 final=0.025 max=0.076 closed=n/a | settle=0.000 (n=100/100) | progress=0.002 net=0.001 yaw_rate_ratio=0.218
      d270: err rms=0.035 initial=0.014 final=0.046 max=0.075 closed=n/a | settle=0.000 (n=100/100) | progress=0.003 net=0.000 yaw_rate_ratio=0.213
      d292: err rms=0.066 initial=0.014 final=0.080 max=0.131 closed=n/a | settle=0.000 (n=99/100) | progress=-0.001 net=-0.001 yaw_rate_ratio=0.334
      d315: err rms=0.045 initial=0.014 final=0.038 max=0.094 closed=n/a | settle=0.000 (n=100/100) | progress=-0.004 net=0.000 yaw_rate_ratio=0.333
      d338: err rms=0.048 initial=0.017 final=0.043 max=0.112 closed=n/a | settle=0.000 (n=100/100) | progress=-0.000 net=0.003 yaw_rate_ratio=0.489

body yaw by hold (window [4 s, hold_s); medians; net = net yaw rate rad/s, ratio = net / wz_cmd SIGNED (the gated turn number), wz_ratio = mean wz / wz_cmd; ripple = std of the 1-s mean wz; drift = |mean wz| on wz = 0 holds; planar = xy drift m/s on pure turns; body = along-bearing speed / |v_cmd|, cross = mean |perp| m/s):
          settle: net=n/a ratio=n/a wz_ratio=n/a ripple=n/a | drift=n/a planar=n/a | body=n/a cross=n/a (n=0)
            d000: net=-0.001 ratio=n/a wz_ratio=n/a ripple=0.054 | drift=0.005 planar=n/a | body=1.007 cross=0.027 (n=100)
            d022: net=-0.001 ratio=n/a wz_ratio=n/a ripple=0.034 | drift=0.005 planar=n/a | body=1.009 cross=0.023 (n=100)
            d045: net=-0.003 ratio=n/a wz_ratio=n/a ripple=0.022 | drift=0.004 planar=n/a | body=1.060 cross=0.038 (n=100)
            d068: net=-0.008 ratio=n/a wz_ratio=n/a ripple=0.026 | drift=0.008 planar=n/a | body=1.051 cross=0.034 (n=100)
            d090: net=-0.001 ratio=n/a wz_ratio=n/a ripple=0.015 | drift=0.003 planar=n/a | body=1.049 cross=0.026 (n=100)
            d112: net=0.005 ratio=n/a wz_ratio=n/a ripple=0.029 | drift=0.005 planar=n/a | body=1.051 cross=0.036 (n=100)
            d135: net=0.001 ratio=n/a wz_ratio=n/a ripple=0.023 | drift=0.004 planar=n/a | body=1.039 cross=0.033 (n=100)
            d158: net=0.006 ratio=n/a wz_ratio=n/a ripple=0.022 | drift=0.007 planar=n/a | body=0.984 cross=0.024 (n=100)
            d180: net=0.003 ratio=n/a wz_ratio=n/a ripple=0.040 | drift=0.004 planar=n/a | body=0.972 cross=0.027 (n=100)
            d202: net=-0.001 ratio=n/a wz_ratio=n/a ripple=0.048 | drift=0.003 planar=n/a | body=0.992 cross=0.031 (n=100)
            d225: net=-0.001 ratio=n/a wz_ratio=n/a ripple=0.036 | drift=0.004 planar=n/a | body=1.020 cross=0.033 (n=100)
            d248: net=0.001 ratio=n/a wz_ratio=n/a ripple=0.024 | drift=0.003 planar=n/a | body=1.053 cross=0.031 (n=100)
            d270: net=0.004 ratio=n/a wz_ratio=n/a ripple=0.022 | drift=0.004 planar=n/a | body=1.050 cross=0.028 (n=100)
            d292: net=0.005 ratio=n/a wz_ratio=n/a ripple=0.042 | drift=0.006 planar=n/a | body=1.035 cross=0.031 (n=100)
            d315: net=0.001 ratio=n/a wz_ratio=n/a ripple=0.042 | drift=0.004 planar=n/a | body=1.034 cross=0.029 (n=100)
            d338: net=-0.000 ratio=n/a wz_ratio=n/a ripple=0.042 | drift=0.004 planar=n/a | body=1.007 cross=0.029 (n=100)

by category (16 waves, 1600 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: n/a):
       d000: n=100 surv=1.000 (d000=1.00) | ratio=1.011 achieved=0.222 tripod=0.692 | closed=n/a progress=-0.001 net=0.002 | drift=0.223 dir_err=0.029
       d022: n=100 surv=1.000 (d022=1.00) | ratio=1.011 achieved=0.222 tripod=0.751 | closed=n/a progress=-0.001 net=0.001 | drift=0.224 dir_err=0.012
       d045: n=100 surv=1.000 (d045=1.00) | ratio=1.054 achieved=0.232 tripod=0.720 | closed=n/a progress=n/a net=0.002 | drift=0.236 dir_err=0.069
       d068: n=100 surv=1.000 (d068=1.00) | ratio=1.047 achieved=0.230 tripod=0.663 | closed=n/a progress=-0.000 net=0.004 | drift=0.235 dir_err=0.031
       d090: n=100 surv=1.000 (d090=1.00) | ratio=1.045 achieved=0.230 tripod=0.662 | closed=n/a progress=n/a net=0.002 | drift=0.233 dir_err=0.009
       d112: n=100 surv=1.000 (d112=1.00) | ratio=1.045 achieved=0.230 tripod=0.648 | closed=n/a progress=-0.003 net=-0.001 | drift=0.235 dir_err=0.031
       d135: n=100 surv=1.000 (d135=1.00) | ratio=1.037 achieved=0.228 tripod=0.648 | closed=n/a progress=n/a net=0.001 | drift=0.231 dir_err=0.057
       d158: n=100 surv=1.000 (d158=1.00) | ratio=0.991 achieved=0.218 tripod=0.767 | closed=n/a progress=-0.004 net=-0.003 | drift=0.220 dir_err=0.015
       d180: n=100 surv=1.000 (d180=1.00) | ratio=0.977 achieved=0.215 tripod=0.686 | closed=n/a progress=-0.001 net=-0.004 | drift=0.216 dir_err=0.036
       d202: n=100 surv=1.000 (d202=1.00) | ratio=0.984 achieved=0.216 tripod=0.667 | closed=n/a progress=-0.001 net=0.000 | drift=0.218 dir_err=0.020
       d225: n=100 surv=1.000 (d225=1.00) | ratio=1.015 achieved=0.223 tripod=0.717 | closed=n/a progress=n/a net=0.002 | drift=0.226 dir_err=0.040
       d248: n=100 surv=1.000 (d248=1.00) | ratio=1.047 achieved=0.230 tripod=0.755 | closed=n/a progress=0.002 net=0.001 | drift=0.231 dir_err=0.057
       d270: n=100 surv=1.000 (d270=1.00) | ratio=1.046 achieved=0.230 tripod=0.757 | closed=n/a progress=0.003 net=0.000 | drift=0.232 dir_err=0.025
       d292: n=100 surv=1.000 (d292=1.00) | ratio=1.029 achieved=0.226 tripod=0.738 | closed=n/a progress=-0.001 net=-0.001 | drift=0.229 dir_err=0.010
       d315: n=100 surv=1.000 (d315=1.00) | ratio=1.024 achieved=0.225 tripod=0.715 | closed=n/a progress=-0.004 net=0.000 | drift=0.227 dir_err=0.011
       d338: n=100 surv=1.000 (d338=1.00) | ratio=1.008 achieved=0.222 tripod=0.655 | closed=n/a progress=-0.000 net=0.003 | drift=0.222 dir_err=0.010

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
       d000: action_jitter=0.111 action_fft_freq=1.118 joint_speed_mean=0.718 rod_speed_mean=0.029 contact_switch_hz=1.778 foot_slip=0.089 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=24.133 slip_per_m=1.201 work_per_m=1194.122 cam_work_per_m=423.451
       d022: action_jitter=0.125 action_fft_freq=1.260 joint_speed_mean=0.671 rod_speed_mean=0.029 contact_switch_hz=1.847 foot_slip=0.082 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=25.014 slip_per_m=1.093 work_per_m=1136.547 cam_work_per_m=300.135
       d045: action_jitter=0.106 action_fft_freq=1.260 joint_speed_mean=0.555 rod_speed_mean=0.029 contact_switch_hz=1.667 foot_slip=0.083 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=21.782 slip_per_m=1.064 work_per_m=1091.276 cam_work_per_m=219.990
       d068: action_jitter=0.091 action_fft_freq=1.249 joint_speed_mean=0.435 rod_speed_mean=0.029 contact_switch_hz=1.569 foot_slip=0.081 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=20.722 slip_per_m=1.064 work_per_m=1099.103 cam_work_per_m=152.446
       d090: action_jitter=0.089 action_fft_freq=1.235 joint_speed_mean=0.379 rod_speed_mean=0.029 contact_switch_hz=1.583 foot_slip=0.075 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=20.760 slip_per_m=1.012 work_per_m=1061.544 cam_work_per_m=141.695
       d112: action_jitter=0.092 action_fft_freq=1.255 joint_speed_mean=0.448 rod_speed_mean=0.029 contact_switch_hz=1.806 foot_slip=0.082 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=23.759 slip_per_m=1.080 work_per_m=1072.442 cam_work_per_m=172.819
       d135: action_jitter=0.106 action_fft_freq=1.231 joint_speed_mean=0.556 rod_speed_mean=0.029 contact_switch_hz=1.889 foot_slip=0.083 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=24.996 slip_per_m=1.056 work_per_m=1044.947 cam_work_per_m=242.696
       d158: action_jitter=0.124 action_fft_freq=1.277 joint_speed_mean=0.655 rod_speed_mean=0.029 contact_switch_hz=1.750 foot_slip=0.080 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=24.135 slip_per_m=1.091 work_per_m=1119.963 cam_work_per_m=283.397
       d180: action_jitter=0.111 action_fft_freq=1.129 joint_speed_mean=0.704 rod_speed_mean=0.029 contact_switch_hz=1.826 foot_slip=0.093 wz_active_frac=0.087 clock_restarts=0.000 swings_per_m=25.747 slip_per_m=1.302 work_per_m=1227.011 cam_work_per_m=422.251
       d202: action_jitter=0.101 action_fft_freq=1.110 joint_speed_mean=0.673 rod_speed_mean=0.029 contact_switch_hz=1.792 foot_slip=0.093 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=25.131 slip_per_m=1.317 work_per_m=1200.806 cam_work_per_m=401.407
       d225: action_jitter=0.091 action_fft_freq=1.107 joint_speed_mean=0.590 rod_speed_mean=0.029 contact_switch_hz=1.750 foot_slip=0.085 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=23.779 slip_per_m=1.179 work_per_m=1139.502 cam_work_per_m=287.382
       d248: action_jitter=0.082 action_fft_freq=1.063 joint_speed_mean=0.460 rod_speed_mean=0.029 contact_switch_hz=1.444 foot_slip=0.076 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=18.999 slip_per_m=1.005 work_per_m=1039.987 cam_work_per_m=250.607
       d270: action_jitter=0.076 action_fft_freq=1.020 joint_speed_mean=0.389 rod_speed_mean=0.028 contact_switch_hz=1.250 foot_slip=0.075 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=16.440 slip_per_m=0.990 work_per_m=1038.879 cam_work_per_m=236.491
       d292: action_jitter=0.084 action_fft_freq=1.078 joint_speed_mean=0.505 rod_speed_mean=0.029 contact_switch_hz=1.590 foot_slip=0.081 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=21.304 slip_per_m=1.078 work_per_m=1120.015 cam_work_per_m=269.279
       d315: action_jitter=0.095 action_fft_freq=1.155 joint_speed_mean=0.609 rod_speed_mean=0.029 contact_switch_hz=1.819 foot_slip=0.085 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=24.177 slip_per_m=1.165 work_per_m=1150.146 cam_work_per_m=324.588
       d338: action_jitter=0.101 action_fft_freq=1.123 joint_speed_mean=0.695 rod_speed_mean=0.029 contact_switch_hz=1.806 foot_slip=0.090 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=24.656 slip_per_m=1.240 work_per_m=1161.795 cam_work_per_m=482.918

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
       d000: steps_per_m=12.427 support_feet=2.977 taps_per_m=11.451 swing_apex=0.116 step_rate_hz=2.750 gait_hz=0.415 clock_hz=0.314 cadence_ratio=1.321 hz_speed_slope=0.383 offset_R=0.057
       d022: steps_per_m=12.952 support_feet=2.962 taps_per_m=11.849 swing_apex=0.122 step_rate_hz=2.833 gait_hz=0.452 clock_hz=0.314 cadence_ratio=1.437 hz_speed_slope=-0.618 offset_R=0.146
       d045: steps_per_m=13.017 support_feet=2.943 taps_per_m=8.232 swing_apex=0.099 step_rate_hz=3.000 gait_hz=0.470 clock_hz=0.314 cadence_ratio=1.495 hz_speed_slope=0.334 offset_R=0.067
       d068: steps_per_m=13.321 support_feet=2.998 taps_per_m=7.052 swing_apex=0.088 step_rate_hz=3.083 gait_hz=0.446 clock_hz=0.314 cadence_ratio=1.418 hz_speed_slope=0.522 offset_R=0.107
       d090: steps_per_m=12.755 support_feet=3.082 taps_per_m=8.105 swing_apex=0.086 step_rate_hz=2.917 gait_hz=0.446 clock_hz=0.314 cadence_ratio=1.418 hz_speed_slope=-0.365 offset_R=0.093
       d112: steps_per_m=13.925 support_feet=3.029 taps_per_m=9.330 swing_apex=0.079 step_rate_hz=3.167 gait_hz=0.470 clock_hz=0.314 cadence_ratio=1.495 hz_speed_slope=-1.255 offset_R=0.093
       d135: steps_per_m=14.396 support_feet=2.920 taps_per_m=10.377 swing_apex=0.091 step_rate_hz=3.250 gait_hz=0.513 clock_hz=0.314 cadence_ratio=1.631 hz_speed_slope=0.490 offset_R=0.108
       d158: steps_per_m=13.006 support_feet=2.995 taps_per_m=10.864 swing_apex=0.114 step_rate_hz=2.833 gait_hz=0.464 clock_hz=0.314 cadence_ratio=1.457 hz_speed_slope=0.483 offset_R=0.075
       d180: steps_per_m=12.915 support_feet=2.972 taps_per_m=12.528 swing_apex=0.110 step_rate_hz=2.750 gait_hz=0.409 clock_hz=0.321 cadence_ratio=1.270 hz_speed_slope=0.761 offset_R=0.076
       d202: steps_per_m=13.827 support_feet=3.004 taps_per_m=11.122 swing_apex=0.120 step_rate_hz=2.917 gait_hz=0.403 clock_hz=0.314 cadence_ratio=1.282 hz_speed_slope=-0.047 offset_R=0.087
       d225: steps_per_m=13.410 support_feet=3.074 taps_per_m=9.819 swing_apex=0.123 step_rate_hz=2.917 gait_hz=0.433 clock_hz=0.314 cadence_ratio=1.379 hz_speed_slope=0.559 offset_R=0.109
       d248: steps_per_m=11.010 support_feet=3.042 taps_per_m=7.863 swing_apex=0.116 step_rate_hz=2.500 gait_hz=0.397 clock_hz=0.314 cadence_ratio=1.262 hz_speed_slope=0.527 offset_R=0.077
       d270: steps_per_m=9.586 support_feet=3.024 taps_per_m=6.738 swing_apex=0.117 step_rate_hz=2.167 gait_hz=0.348 clock_hz=0.314 cadence_ratio=1.107 hz_speed_slope=3.319 offset_R=0.149
       d292: steps_per_m=12.249 support_feet=3.002 taps_per_m=8.862 swing_apex=0.110 step_rate_hz=2.750 gait_hz=0.409 clock_hz=0.314 cadence_ratio=1.300 hz_speed_slope=1.840 offset_R=0.079
       d315: steps_per_m=13.771 support_feet=3.073 taps_per_m=10.445 swing_apex=0.117 step_rate_hz=3.083 gait_hz=0.452 clock_hz=0.314 cadence_ratio=1.437 hz_speed_slope=0.391 offset_R=0.177
       d338: steps_per_m=13.362 support_feet=3.018 taps_per_m=11.229 swing_apex=0.115 step_rate_hz=2.917 gait_hz=0.427 clock_hz=0.314 cadence_ratio=1.359 hz_speed_slope=-0.701 offset_R=0.067

body yaw by category (own hold, window [4 s, hold_s); medians pooled over both hands; per-hand values are in aggregate.suite.by_category[c].by_hand):
       d000: net_yaw_rate=-0.001 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.054 heading_drift=0.005 planar_drift=n/a body_ratio=1.007 cross_body=0.027 adapter_clipped_frac=n/a
       d022: net_yaw_rate=-0.001 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.034 heading_drift=0.005 planar_drift=n/a body_ratio=1.009 cross_body=0.023 adapter_clipped_frac=n/a
       d045: net_yaw_rate=-0.003 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.022 heading_drift=0.004 planar_drift=n/a body_ratio=1.060 cross_body=0.038 adapter_clipped_frac=n/a
       d068: net_yaw_rate=-0.008 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.026 heading_drift=0.008 planar_drift=n/a body_ratio=1.051 cross_body=0.034 adapter_clipped_frac=n/a
       d090: net_yaw_rate=-0.001 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.015 heading_drift=0.003 planar_drift=n/a body_ratio=1.049 cross_body=0.026 adapter_clipped_frac=n/a
       d112: net_yaw_rate=0.005 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.029 heading_drift=0.005 planar_drift=n/a body_ratio=1.051 cross_body=0.036 adapter_clipped_frac=n/a
       d135: net_yaw_rate=0.001 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.023 heading_drift=0.004 planar_drift=n/a body_ratio=1.039 cross_body=0.033 adapter_clipped_frac=n/a
       d158: net_yaw_rate=0.006 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.022 heading_drift=0.007 planar_drift=n/a body_ratio=0.984 cross_body=0.024 adapter_clipped_frac=n/a
       d180: net_yaw_rate=0.003 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.040 heading_drift=0.004 planar_drift=n/a body_ratio=0.972 cross_body=0.027 adapter_clipped_frac=n/a
       d202: net_yaw_rate=-0.001 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.048 heading_drift=0.003 planar_drift=n/a body_ratio=0.992 cross_body=0.031 adapter_clipped_frac=n/a
       d225: net_yaw_rate=-0.001 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.036 heading_drift=0.004 planar_drift=n/a body_ratio=1.020 cross_body=0.033 adapter_clipped_frac=n/a
       d248: net_yaw_rate=0.001 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.024 heading_drift=0.003 planar_drift=n/a body_ratio=1.053 cross_body=0.031 adapter_clipped_frac=n/a
       d270: net_yaw_rate=0.004 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.022 heading_drift=0.004 planar_drift=n/a body_ratio=1.050 cross_body=0.028 adapter_clipped_frac=n/a
       d292: net_yaw_rate=0.005 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.042 heading_drift=0.006 planar_drift=n/a body_ratio=1.035 cross_body=0.031 adapter_clipped_frac=n/a
       d315: net_yaw_rate=0.001 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.042 heading_drift=0.004 planar_drift=n/a body_ratio=1.034 cross_body=0.029 adapter_clipped_frac=n/a
       d338: net_yaw_rate=-0.000 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.042 heading_drift=0.004 planar_drift=n/a body_ratio=1.007 cross_body=0.029 adapter_clipped_frac=n/a

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
