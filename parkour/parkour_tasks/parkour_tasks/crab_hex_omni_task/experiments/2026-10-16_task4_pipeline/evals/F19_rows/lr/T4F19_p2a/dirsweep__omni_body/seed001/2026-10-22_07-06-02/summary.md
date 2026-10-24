```
=== crab-hex gait eval ===
scenario   : dirsweep__omni_body  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-21_23-06-12_seed19_X06_p2a_A15pB/model_19994.pt
episodes   : 1600  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7495163113373637  p25=0.7265051726522901  p75=0.7701908838852233
tippy_tap_fraction median=0.432716049382716
slip_ratio         median=0.21260352348856065
tracking_ratio     median=1.066998035510744  (achieved/commanded vx, walking holds; n=1600)
schedule_completion_rate=1.0
terminations={'schedule_complete': 1600}
facing_error_rms   median=0.035603617233892104  (rad, whole episode; n=1600)

by hold:
    settle: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7750862218696468 (n=1572)
      d000: cmd 0.22 -> achieved 0.236 m/s | tripod median=0.7039435913385267 (n=100)
      d022: cmd 0.22 -> achieved 0.228 m/s | tripod median=0.7472891857100239 (n=100)
      d045: cmd 0.22 -> achieved 0.225 m/s | tripod median=0.7622588767038696 (n=100)
      d068: cmd 0.22 -> achieved 0.233 m/s | tripod median=0.722356197662083 (n=100)
      d090: cmd 0.22 -> achieved 0.235 m/s | tripod median=0.7277945597389395 (n=100)
      d112: cmd 0.22 -> achieved 0.234 m/s | tripod median=0.7239955908376281 (n=100)
      d135: cmd 0.22 -> achieved 0.233 m/s | tripod median=0.7553496137899257 (n=100)
      d158: cmd 0.22 -> achieved 0.235 m/s | tripod median=0.770585281836289 (n=100)
      d180: cmd 0.22 -> achieved 0.233 m/s | tripod median=0.7509539445606452 (n=100)
      d202: cmd 0.22 -> achieved 0.240 m/s | tripod median=0.7621945484695918 (n=100)
      d225: cmd 0.22 -> achieved 0.236 m/s | tripod median=0.7634280451286087 (n=100)
      d248: cmd 0.22 -> achieved 0.237 m/s | tripod median=0.7722916534494616 (n=100)
      d270: cmd 0.22 -> achieved 0.234 m/s | tripod median=0.7950828175203652 (n=100)
      d292: cmd 0.22 -> achieved 0.239 m/s | tripod median=0.7493817915628128 (n=100)
      d315: cmd 0.22 -> achieved 0.238 m/s | tripod median=0.7340904826940208 (n=100)
      d338: cmd 0.22 -> achieved 0.242 m/s | tripod median=0.7281301979154176 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.017 initial=0.005 final=0.014 max=0.034 closed=n/a | settle=0.000 (n=1600/1600) | progress=n/a net=0.001 yaw_rate_ratio=n/a
      d000: err rms=0.059 initial=0.015 final=0.091 max=0.115 closed=n/a | settle=0.000 (n=99/100) | progress=-0.002 net=-0.003 yaw_rate_ratio=0.276
      d022: err rms=0.071 initial=0.017 final=0.092 max=0.132 closed=n/a | settle=0.000 (n=100/100) | progress=-0.002 net=-0.004 yaw_rate_ratio=0.397
      d045: err rms=0.045 initial=0.014 final=0.043 max=0.090 closed=n/a | settle=0.000 (n=100/100) | progress=-0.002 net=-0.000 yaw_rate_ratio=0.220
      d068: err rms=0.021 initial=0.016 final=0.012 max=0.053 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
      d090: err rms=0.025 initial=0.015 final=0.024 max=0.050 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a
      d112: err rms=0.035 initial=0.015 final=0.038 max=0.067 closed=n/a | settle=0.000 (n=100/100) | progress=0.007 net=-0.000 yaw_rate_ratio=0.432
      d135: err rms=0.030 initial=0.014 final=0.023 max=0.068 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.002 yaw_rate_ratio=n/a
      d158: err rms=0.033 initial=0.012 final=0.029 max=0.071 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.002 yaw_rate_ratio=n/a
      d180: err rms=0.033 initial=0.012 final=0.029 max=0.072 closed=n/a | settle=0.000 (n=100/100) | progress=-0.022 net=0.002 yaw_rate_ratio=0.996
      d202: err rms=0.049 initial=0.012 final=0.051 max=0.093 closed=n/a | settle=0.000 (n=100/100) | progress=-0.002 net=0.003 yaw_rate_ratio=0.361
      d225: err rms=0.059 initial=0.013 final=0.062 max=0.104 closed=n/a | settle=0.000 (n=100/100) | progress=0.001 net=0.003 yaw_rate_ratio=0.317
      d248: err rms=0.024 initial=0.013 final=0.020 max=0.058 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
      d270: err rms=0.021 initial=0.014 final=0.022 max=0.045 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a
      d292: err rms=0.038 initial=0.014 final=0.045 max=0.074 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.000 yaw_rate_ratio=n/a
      d315: err rms=0.076 initial=0.014 final=0.084 max=0.124 closed=n/a | settle=0.000 (n=100/100) | progress=-0.000 net=-0.005 yaw_rate_ratio=0.331
      d338: err rms=0.068 initial=0.014 final=0.060 max=0.115 closed=n/a | settle=0.000 (n=100/100) | progress=0.001 net=-0.001 yaw_rate_ratio=0.289

body yaw by hold (window [4 s, hold_s); medians; net = net yaw rate rad/s, ratio = net / wz_cmd SIGNED (the gated turn number), wz_ratio = mean wz / wz_cmd; ripple = std of the 1-s mean wz; drift = |mean wz| on wz = 0 holds; planar = xy drift m/s on pure turns; body = along-bearing speed / |v_cmd|, cross = mean |perp| m/s):
          settle: net=n/a ratio=n/a wz_ratio=n/a ripple=n/a | drift=n/a planar=n/a | body=n/a cross=n/a (n=0)
            d000: net=0.010 ratio=n/a wz_ratio=n/a ripple=0.020 | drift=0.010 planar=n/a | body=1.074 cross=0.026 (n=100)
            d022: net=0.008 ratio=n/a wz_ratio=n/a ripple=0.023 | drift=0.008 planar=n/a | body=1.039 cross=0.029 (n=100)
            d045: net=0.004 ratio=n/a wz_ratio=n/a ripple=0.021 | drift=0.005 planar=n/a | body=1.025 cross=0.026 (n=100)
            d068: net=0.001 ratio=n/a wz_ratio=n/a ripple=0.016 | drift=0.003 planar=n/a | body=1.063 cross=0.025 (n=100)
            d090: net=0.002 ratio=n/a wz_ratio=n/a ripple=0.013 | drift=0.002 planar=n/a | body=1.072 cross=0.022 (n=100)
            d112: net=0.001 ratio=n/a wz_ratio=n/a ripple=0.016 | drift=0.003 planar=n/a | body=1.069 cross=0.025 (n=100)
            d135: net=-0.001 ratio=n/a wz_ratio=n/a ripple=0.022 | drift=0.004 planar=n/a | body=1.063 cross=0.028 (n=100)
            d158: net=-0.001 ratio=n/a wz_ratio=n/a ripple=0.024 | drift=0.006 planar=n/a | body=1.065 cross=0.028 (n=100)
            d180: net=-0.001 ratio=n/a wz_ratio=n/a ripple=0.022 | drift=0.006 planar=n/a | body=1.053 cross=0.023 (n=100)
            d202: net=-0.002 ratio=n/a wz_ratio=n/a ripple=0.020 | drift=0.005 planar=n/a | body=1.095 cross=0.023 (n=100)
            d225: net=-0.001 ratio=n/a wz_ratio=n/a ripple=0.019 | drift=0.003 planar=n/a | body=1.076 cross=0.032 (n=100)
            d248: net=0.000 ratio=n/a wz_ratio=n/a ripple=0.020 | drift=0.002 planar=n/a | body=1.077 cross=0.033 (n=100)
            d270: net=0.001 ratio=n/a wz_ratio=n/a ripple=0.011 | drift=0.003 planar=n/a | body=1.064 cross=0.022 (n=100)
            d292: net=0.001 ratio=n/a wz_ratio=n/a ripple=0.020 | drift=0.003 planar=n/a | body=1.090 cross=0.031 (n=100)
            d315: net=0.001 ratio=n/a wz_ratio=n/a ripple=0.017 | drift=0.003 planar=n/a | body=1.081 cross=0.030 (n=100)
            d338: net=0.001 ratio=n/a wz_ratio=n/a ripple=0.022 | drift=0.004 planar=n/a | body=1.102 cross=0.025 (n=100)

by category (16 waves, 1600 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: n/a):
       d000: n=100 surv=1.000 (d000=1.00) | ratio=1.071 achieved=0.236 tripod=0.704 | closed=n/a progress=-0.002 net=-0.003 | drift=0.234 dir_err=0.066
       d022: n=100 surv=1.000 (d022=1.00) | ratio=1.039 achieved=0.228 tripod=0.747 | closed=n/a progress=-0.002 net=-0.004 | drift=0.229 dir_err=0.073
       d045: n=100 surv=1.000 (d045=1.00) | ratio=1.021 achieved=0.225 tripod=0.762 | closed=n/a progress=-0.002 net=-0.000 | drift=0.227 dir_err=0.017
       d068: n=100 surv=1.000 (d068=1.00) | ratio=1.059 achieved=0.233 tripod=0.722 | closed=n/a progress=n/a net=0.001 | drift=0.233 dir_err=0.018
       d090: n=100 surv=1.000 (d090=1.00) | ratio=1.067 achieved=0.235 tripod=0.728 | closed=n/a progress=n/a net=0.000 | drift=0.234 dir_err=0.009
       d112: n=100 surv=1.000 (d112=1.00) | ratio=1.063 achieved=0.234 tripod=0.724 | closed=n/a progress=0.007 net=-0.000 | drift=0.235 dir_err=0.014
       d135: n=100 surv=1.000 (d135=1.00) | ratio=1.057 achieved=0.233 tripod=0.755 | closed=n/a progress=n/a net=0.002 | drift=0.235 dir_err=0.027
       d158: n=100 surv=1.000 (d158=1.00) | ratio=1.069 achieved=0.235 tripod=0.771 | closed=n/a progress=n/a net=0.002 | drift=0.236 dir_err=0.065
       d180: n=100 surv=1.000 (d180=1.00) | ratio=1.059 achieved=0.233 tripod=0.751 | closed=n/a progress=-0.022 net=0.002 | drift=0.232 dir_err=0.045
       d202: n=100 surv=1.000 (d202=1.00) | ratio=1.093 achieved=0.240 tripod=0.762 | closed=n/a progress=-0.002 net=0.003 | drift=0.239 dir_err=0.025
       d225: n=100 surv=1.000 (d225=1.00) | ratio=1.074 achieved=0.236 tripod=0.763 | closed=n/a progress=0.001 net=0.003 | drift=0.236 dir_err=0.043
       d248: n=100 surv=1.000 (d248=1.00) | ratio=1.078 achieved=0.237 tripod=0.772 | closed=n/a progress=n/a net=0.001 | drift=0.238 dir_err=0.050
       d270: n=100 surv=1.000 (d270=1.00) | ratio=1.065 achieved=0.234 tripod=0.795 | closed=n/a progress=n/a net=0.000 | drift=0.233 dir_err=0.016
       d292: n=100 surv=1.000 (d292=1.00) | ratio=1.085 achieved=0.239 tripod=0.749 | closed=n/a progress=n/a net=-0.000 | drift=0.238 dir_err=0.035
       d315: n=100 surv=1.000 (d315=1.00) | ratio=1.082 achieved=0.238 tripod=0.734 | closed=n/a progress=-0.000 net=-0.005 | drift=0.237 dir_err=0.041
       d338: n=100 surv=1.000 (d338=1.00) | ratio=1.100 achieved=0.242 tripod=0.728 | closed=n/a progress=0.001 net=-0.001 | drift=0.240 dir_err=0.036

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
       d000: action_jitter=0.174 action_fft_freq=1.323 joint_speed_mean=0.727 rod_speed_mean=0.029 contact_switch_hz=1.833 foot_slip=0.098 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=23.691 slip_per_m=1.255 work_per_m=1141.879 cam_work_per_m=411.615
       d022: action_jitter=0.173 action_fft_freq=1.338 joint_speed_mean=0.700 rod_speed_mean=0.029 contact_switch_hz=1.778 foot_slip=0.093 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=23.671 slip_per_m=1.239 work_per_m=1129.458 cam_work_per_m=403.768
       d045: action_jitter=0.171 action_fft_freq=1.448 joint_speed_mean=0.583 rod_speed_mean=0.029 contact_switch_hz=1.840 foot_slip=0.079 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=24.586 slip_per_m=1.077 work_per_m=1064.933 cam_work_per_m=267.501
       d068: action_jitter=0.151 action_fft_freq=1.484 joint_speed_mean=0.446 rod_speed_mean=0.029 contact_switch_hz=1.715 foot_slip=0.079 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=22.253 slip_per_m=1.045 work_per_m=1013.127 cam_work_per_m=243.866
       d090: action_jitter=0.137 action_fft_freq=1.419 joint_speed_mean=0.338 rod_speed_mean=0.029 contact_switch_hz=1.528 foot_slip=0.073 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=19.998 slip_per_m=0.961 work_per_m=991.232 cam_work_per_m=252.144
       d112: action_jitter=0.150 action_fft_freq=1.479 joint_speed_mean=0.430 rod_speed_mean=0.029 contact_switch_hz=1.750 foot_slip=0.078 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=22.685 slip_per_m=1.043 work_per_m=996.943 cam_work_per_m=268.374
       d135: action_jitter=0.156 action_fft_freq=1.382 joint_speed_mean=0.577 rod_speed_mean=0.029 contact_switch_hz=1.701 foot_slip=0.081 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=22.241 slip_per_m=1.073 work_per_m=1008.142 cam_work_per_m=295.806
       d158: action_jitter=0.162 action_fft_freq=1.315 joint_speed_mean=0.684 rod_speed_mean=0.029 contact_switch_hz=1.729 foot_slip=0.098 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=22.295 slip_per_m=1.286 work_per_m=1098.550 cam_work_per_m=402.845
       d180: action_jitter=0.165 action_fft_freq=1.277 joint_speed_mean=0.718 rod_speed_mean=0.029 contact_switch_hz=1.806 foot_slip=0.098 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=23.503 slip_per_m=1.281 work_per_m=1124.006 cam_work_per_m=396.077
       d202: action_jitter=0.167 action_fft_freq=1.298 joint_speed_mean=0.691 rod_speed_mean=0.029 contact_switch_hz=1.917 foot_slip=0.100 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=24.255 slip_per_m=1.253 work_per_m=1109.801 cam_work_per_m=371.961
       d225: action_jitter=0.155 action_fft_freq=1.340 joint_speed_mean=0.598 rod_speed_mean=0.029 contact_switch_hz=1.847 foot_slip=0.091 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=23.824 slip_per_m=1.160 work_per_m=1086.172 cam_work_per_m=300.614
       d248: action_jitter=0.135 action_fft_freq=1.347 joint_speed_mean=0.460 rod_speed_mean=0.029 contact_switch_hz=1.708 foot_slip=0.081 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=21.723 slip_per_m=1.054 work_per_m=1056.651 cam_work_per_m=209.508
       d270: action_jitter=0.134 action_fft_freq=1.425 joint_speed_mean=0.347 rod_speed_mean=0.029 contact_switch_hz=1.590 foot_slip=0.073 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=20.550 slip_per_m=0.976 work_per_m=1046.081 cam_work_per_m=173.133
       d292: action_jitter=0.141 action_fft_freq=1.366 joint_speed_mean=0.460 rod_speed_mean=0.029 contact_switch_hz=1.750 foot_slip=0.084 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=22.286 slip_per_m=1.086 work_per_m=1062.884 cam_work_per_m=212.377
       d315: action_jitter=0.159 action_fft_freq=1.368 joint_speed_mean=0.597 rod_speed_mean=0.029 contact_switch_hz=1.833 foot_slip=0.087 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=23.370 slip_per_m=1.111 work_per_m=1054.268 cam_work_per_m=287.327
       d338: action_jitter=0.173 action_fft_freq=1.332 joint_speed_mean=0.691 rod_speed_mean=0.029 contact_switch_hz=1.917 foot_slip=0.095 wz_active_frac=0.000 clock_restarts=0.000 swings_per_m=24.379 slip_per_m=1.198 work_per_m=1096.523 cam_work_per_m=374.922

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
       d000: steps_per_m=10.798 support_feet=2.955 taps_per_m=12.943 swing_apex=0.123 step_rate_hz=2.500 gait_hz=0.397 clock_hz=0.314 cadence_ratio=1.262 hz_speed_slope=-0.474 offset_R=0.018
       d022: steps_per_m=11.781 support_feet=3.025 taps_per_m=11.739 swing_apex=0.118 step_rate_hz=2.667 gait_hz=0.433 clock_hz=0.314 cadence_ratio=1.365 hz_speed_slope=-0.618 offset_R=0.071
       d045: steps_per_m=13.853 support_feet=3.053 taps_per_m=10.857 swing_apex=0.098 step_rate_hz=3.083 gait_hz=0.513 clock_hz=0.314 cadence_ratio=1.631 hz_speed_slope=0.810 offset_R=0.080
       d068: steps_per_m=14.271 support_feet=3.077 taps_per_m=7.927 swing_apex=0.083 step_rate_hz=3.333 gait_hz=0.555 clock_hz=0.314 cadence_ratio=1.767 hz_speed_slope=1.295 offset_R=0.045
       d090: steps_per_m=12.221 support_feet=3.075 taps_per_m=7.474 swing_apex=0.089 step_rate_hz=2.833 gait_hz=0.482 clock_hz=0.314 cadence_ratio=1.534 hz_speed_slope=-0.389 offset_R=0.113
       d112: steps_per_m=13.413 support_feet=3.072 taps_per_m=9.025 swing_apex=0.082 step_rate_hz=3.083 gait_hz=0.519 clock_hz=0.314 cadence_ratio=1.651 hz_speed_slope=2.510 offset_R=0.098
       d135: steps_per_m=13.321 support_feet=3.053 taps_per_m=8.617 swing_apex=0.095 step_rate_hz=3.083 gait_hz=0.513 clock_hz=0.314 cadence_ratio=1.631 hz_speed_slope=0.483 offset_R=0.006
       d158: steps_per_m=11.404 support_feet=3.043 taps_per_m=10.757 swing_apex=0.122 step_rate_hz=2.667 gait_hz=0.433 clock_hz=0.314 cadence_ratio=1.379 hz_speed_slope=-0.488 offset_R=0.082
       d180: steps_per_m=11.135 support_feet=2.975 taps_per_m=12.256 swing_apex=0.123 step_rate_hz=2.583 gait_hz=0.403 clock_hz=0.314 cadence_ratio=1.282 hz_speed_slope=-0.512 offset_R=0.131
       d202: steps_per_m=11.377 support_feet=2.987 taps_per_m=12.716 swing_apex=0.118 step_rate_hz=2.667 gait_hz=0.427 clock_hz=0.314 cadence_ratio=1.359 hz_speed_slope=-0.042 offset_R=0.097
       d225: steps_per_m=12.499 support_feet=2.995 taps_per_m=11.127 swing_apex=0.103 step_rate_hz=2.917 gait_hz=0.476 clock_hz=0.314 cadence_ratio=1.513 hz_speed_slope=0.061 offset_R=0.051
       d248: steps_per_m=13.392 support_feet=3.060 taps_per_m=7.969 swing_apex=0.082 step_rate_hz=3.167 gait_hz=0.531 clock_hz=0.314 cadence_ratio=1.690 hz_speed_slope=0.367 offset_R=0.155
       d270: steps_per_m=13.489 support_feet=3.101 taps_per_m=6.958 swing_apex=0.079 step_rate_hz=3.083 gait_hz=0.531 clock_hz=0.314 cadence_ratio=1.690 hz_speed_slope=3.483 offset_R=0.058
       d292: steps_per_m=13.341 support_feet=3.072 taps_per_m=8.862 swing_apex=0.086 step_rate_hz=3.167 gait_hz=0.513 clock_hz=0.314 cadence_ratio=1.631 hz_speed_slope=3.609 offset_R=0.067
       d315: steps_per_m=12.704 support_feet=3.013 taps_per_m=10.644 swing_apex=0.102 step_rate_hz=3.000 gait_hz=0.473 clock_hz=0.314 cadence_ratio=1.499 hz_speed_slope=0.661 offset_R=0.017
       d338: steps_per_m=11.529 support_feet=3.002 taps_per_m=12.668 swing_apex=0.114 step_rate_hz=2.750 gait_hz=0.421 clock_hz=0.314 cadence_ratio=1.340 hz_speed_slope=-0.203 offset_R=0.138

body yaw by category (own hold, window [4 s, hold_s); medians pooled over both hands; per-hand values are in aggregate.suite.by_category[c].by_hand):
       d000: net_yaw_rate=0.010 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.020 heading_drift=0.010 planar_drift=n/a body_ratio=1.074 cross_body=0.026 adapter_clipped_frac=n/a
       d022: net_yaw_rate=0.008 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.023 heading_drift=0.008 planar_drift=n/a body_ratio=1.039 cross_body=0.029 adapter_clipped_frac=n/a
       d045: net_yaw_rate=0.004 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.021 heading_drift=0.005 planar_drift=n/a body_ratio=1.025 cross_body=0.026 adapter_clipped_frac=n/a
       d068: net_yaw_rate=0.001 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.016 heading_drift=0.003 planar_drift=n/a body_ratio=1.063 cross_body=0.025 adapter_clipped_frac=n/a
       d090: net_yaw_rate=0.002 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.013 heading_drift=0.002 planar_drift=n/a body_ratio=1.072 cross_body=0.022 adapter_clipped_frac=n/a
       d112: net_yaw_rate=0.001 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.016 heading_drift=0.003 planar_drift=n/a body_ratio=1.069 cross_body=0.025 adapter_clipped_frac=n/a
       d135: net_yaw_rate=-0.001 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.022 heading_drift=0.004 planar_drift=n/a body_ratio=1.063 cross_body=0.028 adapter_clipped_frac=n/a
       d158: net_yaw_rate=-0.001 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.024 heading_drift=0.006 planar_drift=n/a body_ratio=1.065 cross_body=0.028 adapter_clipped_frac=n/a
       d180: net_yaw_rate=-0.001 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.022 heading_drift=0.006 planar_drift=n/a body_ratio=1.053 cross_body=0.023 adapter_clipped_frac=n/a
       d202: net_yaw_rate=-0.002 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.020 heading_drift=0.005 planar_drift=n/a body_ratio=1.095 cross_body=0.023 adapter_clipped_frac=n/a
       d225: net_yaw_rate=-0.001 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.019 heading_drift=0.003 planar_drift=n/a body_ratio=1.076 cross_body=0.032 adapter_clipped_frac=n/a
       d248: net_yaw_rate=0.000 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.020 heading_drift=0.002 planar_drift=n/a body_ratio=1.077 cross_body=0.033 adapter_clipped_frac=n/a
       d270: net_yaw_rate=0.001 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.011 heading_drift=0.003 planar_drift=n/a body_ratio=1.064 cross_body=0.022 adapter_clipped_frac=n/a
       d292: net_yaw_rate=0.001 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.020 heading_drift=0.003 planar_drift=n/a body_ratio=1.090 cross_body=0.031 adapter_clipped_frac=n/a
       d315: net_yaw_rate=0.001 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.017 heading_drift=0.003 planar_drift=n/a body_ratio=1.081 cross_body=0.030 adapter_clipped_frac=n/a
       d338: net_yaw_rate=0.001 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.022 heading_drift=0.004 planar_drift=n/a body_ratio=1.102 cross_body=0.025 adapter_clipped_frac=n/a

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
