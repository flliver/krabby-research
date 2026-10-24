```
=== crab-hex gait eval ===
scenario   : grid__omni_body  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_28000.pt
episodes   : 2000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6749450422990129  p25=0.6204724933181753  p75=0.7273847343133694
tippy_tap_fraction median=0.43023255813953487
slip_ratio         median=0.22572111217706664
tracking_ratio     median=0.8730072663580033  (achieved/commanded vx, walking holds; n=1600)
schedule_completion_rate=1.0
terminations={'schedule_complete': 2000}
facing_error_rms   median=0.22774080764031318  (rad, whole episode; n=2000)

by hold:
    settle: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.6595841293772794 (n=1905)
   g_v0_w0: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.7607924935666929 (n=100)
  g_v0_w0p08: cmd 0.00 -> achieved -0.002 m/s | tripod median=0.762703600725233 (n=100)
  g_v0_w0p15: cmd 0.00 -> achieved -0.002 m/s | tripod median=0.7637917447820696 (n=100)
  g_v0_w0p3: cmd 0.00 -> achieved -0.011 m/s | tripod median=0.774805180186489 (n=100)
  g_v0p22_fwd_w0: cmd 0.22 -> achieved 0.223 m/s | tripod median=0.690727143185423 (n=100)
  g_v0p22_fwd_w0p08: cmd 0.22 -> achieved 0.192 m/s | tripod median=0.6796196280124551 (n=100)
  g_v0p22_fwd_w0p15: cmd 0.22 -> achieved 0.169 m/s | tripod median=0.6567351652213447 (n=100)
  g_v0p22_fwd_w0p3: cmd 0.22 -> achieved 0.157 m/s | tripod median=0.6517750908006463 (n=100)
  g_v0p22_lat_w0: cmd 0.22 -> achieved 0.229 m/s | tripod median=0.6664653886480788 (n=100)
  g_v0p22_lat_w0p08: cmd 0.22 -> achieved 0.218 m/s | tripod median=0.6471690972276776 (n=100)
  g_v0p22_lat_w0p15: cmd 0.22 -> achieved 0.211 m/s | tripod median=0.6789145859855854 (n=100)
  g_v0p22_lat_w0p3: cmd 0.22 -> achieved 0.197 m/s | tripod median=0.7579264558658471 (n=100)
  g_v0p3_fwd_w0: cmd 0.30 -> achieved 0.278 m/s | tripod median=0.5941638882926172 (n=100)
  g_v0p3_fwd_w0p08: cmd 0.30 -> achieved 0.234 m/s | tripod median=0.471066926006669 (n=100)
  g_v0p3_fwd_w0p15: cmd 0.30 -> achieved 0.216 m/s | tripod median=0.437040254635522 (n=100)
  g_v0p3_fwd_w0p3: cmd 0.30 -> achieved 0.207 m/s | tripod median=0.42399013142554065 (n=100)
  g_v0p3_lat_w0: cmd 0.30 -> achieved 0.277 m/s | tripod median=0.6725024963695831 (n=100)
  g_v0p3_lat_w0p08: cmd 0.30 -> achieved 0.264 m/s | tripod median=0.6767660855752458 (n=100)
  g_v0p3_lat_w0p15: cmd 0.30 -> achieved 0.254 m/s | tripod median=0.7282820055398904 (n=100)
  g_v0p3_lat_w0p3: cmd 0.30 -> achieved 0.236 m/s | tripod median=0.7314239228898115 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.015 initial=0.008 final=0.017 max=0.031 closed=n/a | settle=0.000 (n=2000/2000) | progress=n/a net=-0.000 yaw_rate_ratio=n/a
   g_v0_w0: err rms=0.033 initial=0.017 final=0.034 max=0.063 closed=n/a | settle=0.000 (n=100/100) | progress=-0.003 net=-0.000 yaw_rate_ratio=0.657
  g_v0_w0p08: err rms=0.138 initial=0.018 final=0.160 max=0.223 closed=n/a | settle=11.780 (n=38/100) | progress=0.076 net=0.013 yaw_rate_ratio=0.618
  g_v0_w0p15: err rms=0.218 initial=0.019 final=0.256 max=0.323 closed=n/a | settle=11.810 (n=4/100) | progress=0.140 net=0.019 yaw_rate_ratio=0.601
  g_v0_w0p3: err rms=0.532 initial=0.016 final=0.600 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.180 net=0.049 yaw_rate_ratio=0.669
  g_v0p22_fwd_w0: err rms=0.051 initial=0.016 final=0.040 max=0.124 closed=n/a | settle=0.000 (n=100/100) | progress=-0.000 net=0.004 yaw_rate_ratio=0.637
  g_v0p22_fwd_w0p08: err rms=0.256 initial=0.015 final=0.401 max=0.410 closed=n/a | settle=n/a (n=0/100) | progress=0.053 net=0.030 yaw_rate_ratio=0.301
  g_v0p22_fwd_w0p15: err rms=0.487 initial=0.016 final=0.600 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.061 net=0.051 yaw_rate_ratio=0.327
  g_v0p22_fwd_w0p3: err rms=0.556 initial=0.016 final=0.600 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.063 net=0.050 yaw_rate_ratio=0.337
  g_v0p22_lat_w0: err rms=0.020 initial=0.019 final=0.017 max=0.047 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.003 yaw_rate_ratio=n/a
  g_v0p22_lat_w0p08: err rms=0.155 initial=0.016 final=0.148 max=0.239 closed=n/a | settle=11.740 (n=52/100) | progress=0.079 net=0.010 yaw_rate_ratio=0.542
  g_v0p22_lat_w0p15: err rms=0.257 initial=0.015 final=0.271 max=0.357 closed=n/a | settle=n/a (n=0/100) | progress=0.138 net=0.021 yaw_rate_ratio=0.564
  g_v0p22_lat_w0p3: err rms=0.538 initial=0.016 final=0.600 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.170 net=0.050 yaw_rate_ratio=0.630
  g_v0p3_fwd_w0: err rms=0.044 initial=0.017 final=0.077 max=0.093 closed=n/a | settle=0.000 (n=96/100) | progress=-0.009 net=0.006 yaw_rate_ratio=0.423
  g_v0p3_fwd_w0p08: err rms=0.395 initial=0.016 final=0.597 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.033 net=0.050 yaw_rate_ratio=0.363
  g_v0p3_fwd_w0p15: err rms=0.512 initial=0.016 final=0.600 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.042 net=0.050 yaw_rate_ratio=0.354
  g_v0p3_fwd_w0p3: err rms=0.562 initial=0.017 final=0.600 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.044 net=0.051 yaw_rate_ratio=0.347
  g_v0p3_lat_w0: err rms=0.018 initial=0.016 final=0.013 max=0.042 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.002 yaw_rate_ratio=n/a
  g_v0p3_lat_w0p08: err rms=0.155 initial=0.017 final=0.157 max=0.233 closed=n/a | settle=11.800 (n=39/100) | progress=0.078 net=0.013 yaw_rate_ratio=0.533
  g_v0p3_lat_w0p15: err rms=0.306 initial=0.017 final=0.392 max=0.428 closed=n/a | settle=n/a (n=0/100) | progress=0.127 net=0.031 yaw_rate_ratio=0.536
  g_v0p3_lat_w0p3: err rms=0.545 initial=0.016 final=0.600 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.152 net=0.051 yaw_rate_ratio=0.576

body yaw by hold (window [4 s, hold_s); medians; net = net yaw rate rad/s, ratio = net / wz_cmd SIGNED (the gated turn number), wz_ratio = mean wz / wz_cmd; ripple = std of the 1-s mean wz; drift = |mean wz| on wz = 0 holds; planar = xy drift m/s on pure turns; body = along-bearing speed / |v_cmd|, cross = mean |perp| m/s):
          settle: net=n/a ratio=n/a wz_ratio=n/a ripple=n/a | drift=n/a planar=n/a | body=n/a cross=n/a (n=0)
         g_v0_w0: net=0.002 ratio=n/a wz_ratio=n/a ripple=0.016 | drift=0.003 planar=n/a | body=n/a cross=n/a (n=100)
      g_v0_w0p08: net=0.077 ratio=0.964 wz_ratio=0.971 ripple=0.037 | drift=n/a planar=0.010 | body=n/a cross=n/a (n=100)
      g_v0_w0p15: net=0.144 ratio=0.959 wz_ratio=0.964 ripple=0.033 | drift=n/a planar=0.009 | body=n/a cross=n/a (n=100)
       g_v0_w0p3: net=0.184 ratio=0.613 wz_ratio=0.615 ripple=0.027 | drift=n/a planar=0.017 | body=n/a cross=n/a (n=100)
  g_v0p22_fwd_w0: net=0.000 ratio=n/a wz_ratio=n/a ripple=0.058 | drift=0.005 planar=n/a | body=1.012 cross=0.026 (n=100)
  g_v0p22_fwd_w0p08: net=0.052 ratio=0.645 wz_ratio=0.640 ripple=0.032 | drift=n/a planar=n/a | body=0.832 cross=0.026 (n=100)
  g_v0p22_fwd_w0p15: net=0.059 ratio=0.392 wz_ratio=0.384 ripple=0.040 | drift=n/a planar=n/a | body=0.708 cross=0.027 (n=100)
  g_v0p22_fwd_w0p3: net=0.057 ratio=0.191 wz_ratio=0.187 ripple=0.030 | drift=n/a planar=n/a | body=0.680 cross=0.028 (n=100)
  g_v0p22_lat_w0: net=-0.002 ratio=n/a wz_ratio=n/a ripple=0.016 | drift=0.003 planar=n/a | body=1.043 cross=0.026 (n=100)
  g_v0p22_lat_w0p08: net=0.085 ratio=1.064 wz_ratio=1.071 ripple=0.029 | drift=n/a planar=n/a | body=0.977 cross=0.041 (n=100)
  g_v0p22_lat_w0p15: net=0.153 ratio=1.023 wz_ratio=1.029 ripple=0.023 | drift=n/a planar=n/a | body=0.944 cross=0.040 (n=100)
  g_v0p22_lat_w0p3: net=0.180 ratio=0.601 wz_ratio=0.603 ripple=0.029 | drift=n/a planar=n/a | body=0.882 cross=0.051 (n=100)
   g_v0p3_fwd_w0: net=-0.005 ratio=n/a wz_ratio=n/a ripple=0.026 | drift=0.005 planar=n/a | body=0.915 cross=0.026 (n=100)
  g_v0p3_fwd_w0p08: net=0.037 ratio=0.464 wz_ratio=0.458 ripple=0.032 | drift=n/a planar=n/a | body=0.741 cross=0.032 (n=100)
  g_v0p3_fwd_w0p15: net=0.045 ratio=0.302 wz_ratio=0.295 ripple=0.029 | drift=n/a planar=n/a | body=0.678 cross=0.033 (n=100)
  g_v0p3_fwd_w0p3: net=0.042 ratio=0.141 wz_ratio=0.138 ripple=0.029 | drift=n/a planar=n/a | body=0.654 cross=0.034 (n=100)
   g_v0p3_lat_w0: net=0.000 ratio=n/a wz_ratio=n/a ripple=0.013 | drift=0.002 planar=n/a | body=0.924 cross=0.030 (n=100)
  g_v0p3_lat_w0p08: net=0.083 ratio=1.042 wz_ratio=1.047 ripple=0.026 | drift=n/a planar=n/a | body=0.872 cross=0.040 (n=100)
  g_v0p3_lat_w0p15: net=0.141 ratio=0.939 wz_ratio=0.942 ripple=0.025 | drift=n/a planar=n/a | body=0.829 cross=0.039 (n=100)
  g_v0p3_lat_w0p3: net=0.163 ratio=0.543 wz_ratio=0.544 ripple=0.032 | drift=n/a planar=n/a | body=0.766 cross=0.056 (n=100)

by category (20 waves, 2000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: n/a):
    g_v0_w0: n=100 surv=1.000 (g_v0_w0=1.00) | ratio=n/a achieved=0.002 tripod=0.761 | closed=n/a progress=-0.003 net=-0.000 | drift=0.035 dir_err=n/a
  g_v0_w0p08: n=100 surv=1.000 (g_v0_w0p08=1.00) | ratio=n/a achieved=-0.002 tripod=0.763 | closed=n/a progress=0.076 net=0.013 | drift=0.041 dir_err=n/a
  g_v0_w0p15: n=100 surv=1.000 (g_v0_w0p15=1.00) | ratio=n/a achieved=-0.002 tripod=0.764 | closed=n/a progress=0.140 net=0.019 | drift=0.043 dir_err=n/a
  g_v0_w0p3: n=100 surv=1.000 (g_v0_w0p3=1.00) | ratio=n/a achieved=-0.011 tripod=0.775 | closed=n/a progress=0.180 net=0.049 | drift=0.053 dir_err=n/a
  g_v0p22_fwd_w0: n=100 surv=1.000 (g_v0p22_fwd_w0=1.00) | ratio=1.013 achieved=0.223 tripod=0.691 | closed=n/a progress=-0.000 net=0.004 | drift=0.223 dir_err=0.029
  g_v0p22_fwd_w0p08: n=100 surv=1.000 (g_v0p22_fwd_w0p08=1.00) | ratio=0.873 achieved=0.192 tripod=0.680 | closed=n/a progress=0.053 net=0.030 | drift=0.195 dir_err=0.017
  g_v0p22_fwd_w0p15: n=100 surv=1.000 (g_v0p22_fwd_w0p15=1.00) | ratio=0.766 achieved=0.169 tripod=0.657 | closed=n/a progress=0.061 net=0.051 | drift=0.175 dir_err=0.031
  g_v0p22_fwd_w0p3: n=100 surv=1.000 (g_v0p22_fwd_w0p3=1.00) | ratio=0.716 achieved=0.157 tripod=0.652 | closed=n/a progress=0.063 net=0.050 | drift=0.165 dir_err=0.040
  g_v0p22_lat_w0: n=100 surv=1.000 (g_v0p22_lat_w0=1.00) | ratio=1.042 achieved=0.229 tripod=0.666 | closed=n/a progress=n/a net=0.003 | drift=0.232 dir_err=0.009
  g_v0p22_lat_w0p08: n=100 surv=1.000 (g_v0p22_lat_w0p08=1.00) | ratio=0.991 achieved=0.218 tripod=0.647 | closed=n/a progress=0.079 net=0.010 | drift=0.224 dir_err=0.039
  g_v0p22_lat_w0p15: n=100 surv=1.000 (g_v0p22_lat_w0p15=1.00) | ratio=0.961 achieved=0.211 tripod=0.679 | closed=n/a progress=0.138 net=0.021 | drift=0.218 dir_err=0.030
  g_v0p22_lat_w0p3: n=100 surv=1.000 (g_v0p22_lat_w0p3=1.00) | ratio=0.895 achieved=0.197 tripod=0.758 | closed=n/a progress=0.170 net=0.050 | drift=0.211 dir_err=0.016
  g_v0p3_fwd_w0: n=100 surv=1.000 (g_v0p3_fwd_w0=1.00) | ratio=0.925 achieved=0.278 tripod=0.594 | closed=n/a progress=-0.009 net=0.006 | drift=0.278 dir_err=0.011
  g_v0p3_fwd_w0p08: n=100 surv=1.000 (g_v0p3_fwd_w0p08=1.00) | ratio=0.780 achieved=0.234 tripod=0.471 | closed=n/a progress=0.033 net=0.050 | drift=0.240 dir_err=0.042
  g_v0p3_fwd_w0p15: n=100 surv=1.000 (g_v0p3_fwd_w0p15=1.00) | ratio=0.721 achieved=0.216 tripod=0.437 | closed=n/a progress=0.042 net=0.050 | drift=0.225 dir_err=0.052
  g_v0p3_fwd_w0p3: n=100 surv=1.000 (g_v0p3_fwd_w0p3=1.00) | ratio=0.691 achieved=0.207 tripod=0.424 | closed=n/a progress=0.044 net=0.051 | drift=0.217 dir_err=0.053
  g_v0p3_lat_w0: n=100 surv=1.000 (g_v0p3_lat_w0=1.00) | ratio=0.924 achieved=0.277 tripod=0.673 | closed=n/a progress=n/a net=0.002 | drift=0.279 dir_err=0.009
  g_v0p3_lat_w0p08: n=100 surv=1.000 (g_v0p3_lat_w0p08=1.00) | ratio=0.878 achieved=0.264 tripod=0.677 | closed=n/a progress=0.078 net=0.013 | drift=0.268 dir_err=0.044
  g_v0p3_lat_w0p15: n=100 surv=1.000 (g_v0p3_lat_w0p15=1.00) | ratio=0.846 achieved=0.254 tripod=0.728 | closed=n/a progress=0.127 net=0.031 | drift=0.260 dir_err=0.041
  g_v0p3_lat_w0p3: n=100 surv=1.000 (g_v0p3_lat_w0p3=1.00) | ratio=0.786 achieved=0.236 tripod=0.731 | closed=n/a progress=0.152 net=0.051 | drift=0.250 dir_err=0.028

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
    g_v0_w0: 
  g_v0_w0p08: 
  g_v0_w0p15: 
  g_v0_w0p3: 
  g_v0p22_fwd_w0: 
  g_v0p22_fwd_w0p08: 
  g_v0p22_fwd_w0p15: 
  g_v0p22_fwd_w0p3: 
  g_v0p22_lat_w0: 
  g_v0p22_lat_w0p08: 
  g_v0p22_lat_w0p15: 
  g_v0p22_lat_w0p3: 
  g_v0p3_fwd_w0: 
  g_v0p3_fwd_w0p08: 
  g_v0p3_fwd_w0p15: 
  g_v0p3_fwd_w0p3: 
  g_v0p3_lat_w0: 
  g_v0p3_lat_w0p08: 
  g_v0p3_lat_w0p15: 
  g_v0p3_lat_w0p3: 

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
    g_v0_w0: 
  g_v0_w0p08: 
  g_v0_w0p15: 
  g_v0_w0p3: 
  g_v0p22_fwd_w0: 
  g_v0p22_fwd_w0p08: 
  g_v0p22_fwd_w0p15: 
  g_v0p22_fwd_w0p3: 
  g_v0p22_lat_w0: 
  g_v0p22_lat_w0p08: 
  g_v0p22_lat_w0p15: 
  g_v0p22_lat_w0p3: 
  g_v0p3_fwd_w0: 
  g_v0p3_fwd_w0p08: 
  g_v0p3_fwd_w0p15: 
  g_v0p3_fwd_w0p3: 
  g_v0p3_lat_w0: 
  g_v0p3_lat_w0p08: 
  g_v0p3_lat_w0p15: 
  g_v0p3_lat_w0p3: 

body yaw by category (own hold, window [4 s, hold_s); medians pooled over both hands; per-hand values are in aggregate.suite.by_category[c].by_hand):
    g_v0_w0: net_yaw_rate=0.002 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.016 heading_drift=0.003 planar_drift=n/a body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  g_v0_w0p08: net_yaw_rate=0.077 net_yaw_ratio=0.964 wz_ratio=0.971 yaw_ripple=0.037 heading_drift=n/a planar_drift=0.010 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  g_v0_w0p15: net_yaw_rate=0.144 net_yaw_ratio=0.959 wz_ratio=0.964 yaw_ripple=0.033 heading_drift=n/a planar_drift=0.009 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  g_v0_w0p3: net_yaw_rate=0.184 net_yaw_ratio=0.613 wz_ratio=0.615 yaw_ripple=0.027 heading_drift=n/a planar_drift=0.017 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  g_v0p22_fwd_w0: net_yaw_rate=0.000 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.058 heading_drift=0.005 planar_drift=n/a body_ratio=1.012 cross_body=0.026 adapter_clipped_frac=n/a
  g_v0p22_fwd_w0p08: net_yaw_rate=0.052 net_yaw_ratio=0.645 wz_ratio=0.640 yaw_ripple=0.032 heading_drift=n/a planar_drift=n/a body_ratio=0.832 cross_body=0.026 adapter_clipped_frac=n/a
  g_v0p22_fwd_w0p15: net_yaw_rate=0.059 net_yaw_ratio=0.392 wz_ratio=0.384 yaw_ripple=0.040 heading_drift=n/a planar_drift=n/a body_ratio=0.708 cross_body=0.027 adapter_clipped_frac=n/a
  g_v0p22_fwd_w0p3: net_yaw_rate=0.057 net_yaw_ratio=0.191 wz_ratio=0.187 yaw_ripple=0.030 heading_drift=n/a planar_drift=n/a body_ratio=0.680 cross_body=0.028 adapter_clipped_frac=n/a
  g_v0p22_lat_w0: net_yaw_rate=-0.002 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.016 heading_drift=0.003 planar_drift=n/a body_ratio=1.043 cross_body=0.026 adapter_clipped_frac=n/a
  g_v0p22_lat_w0p08: net_yaw_rate=0.085 net_yaw_ratio=1.064 wz_ratio=1.071 yaw_ripple=0.029 heading_drift=n/a planar_drift=n/a body_ratio=0.977 cross_body=0.041 adapter_clipped_frac=n/a
  g_v0p22_lat_w0p15: net_yaw_rate=0.153 net_yaw_ratio=1.023 wz_ratio=1.029 yaw_ripple=0.023 heading_drift=n/a planar_drift=n/a body_ratio=0.944 cross_body=0.040 adapter_clipped_frac=n/a
  g_v0p22_lat_w0p3: net_yaw_rate=0.180 net_yaw_ratio=0.601 wz_ratio=0.603 yaw_ripple=0.029 heading_drift=n/a planar_drift=n/a body_ratio=0.882 cross_body=0.051 adapter_clipped_frac=n/a
  g_v0p3_fwd_w0: net_yaw_rate=-0.005 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.026 heading_drift=0.005 planar_drift=n/a body_ratio=0.915 cross_body=0.026 adapter_clipped_frac=n/a
  g_v0p3_fwd_w0p08: net_yaw_rate=0.037 net_yaw_ratio=0.464 wz_ratio=0.458 yaw_ripple=0.032 heading_drift=n/a planar_drift=n/a body_ratio=0.741 cross_body=0.032 adapter_clipped_frac=n/a
  g_v0p3_fwd_w0p15: net_yaw_rate=0.045 net_yaw_ratio=0.302 wz_ratio=0.295 yaw_ripple=0.029 heading_drift=n/a planar_drift=n/a body_ratio=0.678 cross_body=0.033 adapter_clipped_frac=n/a
  g_v0p3_fwd_w0p3: net_yaw_rate=0.042 net_yaw_ratio=0.141 wz_ratio=0.138 yaw_ripple=0.029 heading_drift=n/a planar_drift=n/a body_ratio=0.654 cross_body=0.034 adapter_clipped_frac=n/a
  g_v0p3_lat_w0: net_yaw_rate=0.000 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.013 heading_drift=0.002 planar_drift=n/a body_ratio=0.924 cross_body=0.030 adapter_clipped_frac=n/a
  g_v0p3_lat_w0p08: net_yaw_rate=0.083 net_yaw_ratio=1.042 wz_ratio=1.047 yaw_ripple=0.026 heading_drift=n/a planar_drift=n/a body_ratio=0.872 cross_body=0.040 adapter_clipped_frac=n/a
  g_v0p3_lat_w0p15: net_yaw_rate=0.141 net_yaw_ratio=0.939 wz_ratio=0.942 yaw_ripple=0.025 heading_drift=n/a planar_drift=n/a body_ratio=0.829 cross_body=0.039 adapter_clipped_frac=n/a
  g_v0p3_lat_w0p3: net_yaw_rate=0.163 net_yaw_ratio=0.543 wz_ratio=0.544 yaw_ripple=0.032 heading_drift=n/a planar_drift=n/a body_ratio=0.766 cross_body=0.056 adapter_clipped_frac=n/a

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
