```
=== crab-hex gait eval ===
scenario   : grid__omni_body  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_28000.pt
episodes   : 2000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6786742005449229  p25=0.622976018548252  p75=0.7288440854292001
tippy_tap_fraction median=0.42857142857142855
slip_ratio         median=0.22986441889531894
tracking_ratio     median=0.8745507494675294  (achieved/commanded vx, walking holds; n=1600)
schedule_completion_rate=1.0
terminations={'schedule_complete': 2000}
facing_error_rms   median=0.2270174038499754  (rad, whole episode; n=2000)

by hold:
    settle: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.6772137098174271 (n=1864)
   g_v0_w0: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7616158487486702 (n=100)
  g_v0_w0p08: cmd 0.00 -> achieved -0.002 m/s | tripod median=0.7598286625700785 (n=100)
  g_v0_w0p15: cmd 0.00 -> achieved -0.003 m/s | tripod median=0.7696280882273172 (n=100)
  g_v0_w0p3: cmd 0.00 -> achieved -0.010 m/s | tripod median=0.7783152925696082 (n=100)
  g_v0p22_fwd_w0: cmd 0.22 -> achieved 0.223 m/s | tripod median=0.6903746428145292 (n=100)
  g_v0p22_fwd_w0p08: cmd 0.22 -> achieved 0.193 m/s | tripod median=0.6938103713090916 (n=100)
  g_v0p22_fwd_w0p15: cmd 0.22 -> achieved 0.167 m/s | tripod median=0.662364844975993 (n=100)
  g_v0p22_fwd_w0p3: cmd 0.22 -> achieved 0.157 m/s | tripod median=0.651113897555055 (n=100)
  g_v0p22_lat_w0: cmd 0.22 -> achieved 0.230 m/s | tripod median=0.6643807853134591 (n=100)
  g_v0p22_lat_w0p08: cmd 0.22 -> achieved 0.219 m/s | tripod median=0.66124477414021 (n=100)
  g_v0p22_lat_w0p15: cmd 0.22 -> achieved 0.212 m/s | tripod median=0.6821932769145812 (n=100)
  g_v0p22_lat_w0p3: cmd 0.22 -> achieved 0.196 m/s | tripod median=0.7573654089534962 (n=100)
  g_v0p3_fwd_w0: cmd 0.30 -> achieved 0.277 m/s | tripod median=0.5944427054298066 (n=100)
  g_v0p3_fwd_w0p08: cmd 0.30 -> achieved 0.235 m/s | tripod median=0.46223424732621443 (n=100)
  g_v0p3_fwd_w0p15: cmd 0.30 -> achieved 0.218 m/s | tripod median=0.45019689088783643 (n=100)
  g_v0p3_fwd_w0p3: cmd 0.30 -> achieved 0.207 m/s | tripod median=0.4232866020638232 (n=100)
  g_v0p3_lat_w0: cmd 0.30 -> achieved 0.277 m/s | tripod median=0.6740350213031885 (n=100)
  g_v0p3_lat_w0p08: cmd 0.30 -> achieved 0.265 m/s | tripod median=0.6850861615055108 (n=100)
  g_v0p3_lat_w0p15: cmd 0.30 -> achieved 0.253 m/s | tripod median=0.717855417267149 (n=100)
  g_v0p3_lat_w0p3: cmd 0.30 -> achieved 0.235 m/s | tripod median=0.737472487033977 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.014 initial=0.008 final=0.015 max=0.029 closed=n/a | settle=0.000 (n=2000/2000) | progress=n/a net=0.000 yaw_rate_ratio=n/a
   g_v0_w0: err rms=0.031 initial=0.014 final=0.032 max=0.061 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a
  g_v0_w0p08: err rms=0.138 initial=0.015 final=0.159 max=0.223 closed=n/a | settle=11.680 (n=44/100) | progress=0.076 net=0.013 yaw_rate_ratio=0.613
  g_v0_w0p15: err rms=0.221 initial=0.014 final=0.259 max=0.332 closed=n/a | settle=11.940 (n=1/100) | progress=0.140 net=0.019 yaw_rate_ratio=0.607
  g_v0_w0p3: err rms=0.532 initial=0.014 final=0.600 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.180 net=0.049 yaw_rate_ratio=0.672
  g_v0p22_fwd_w0: err rms=0.053 initial=0.014 final=0.055 max=0.124 closed=n/a | settle=0.000 (n=99/100) | progress=-0.001 net=0.001 yaw_rate_ratio=0.640
  g_v0p22_fwd_w0p08: err rms=0.255 initial=0.015 final=0.401 max=0.405 closed=n/a | settle=n/a (n=0/100) | progress=0.052 net=0.031 yaw_rate_ratio=0.295
  g_v0p22_fwd_w0p15: err rms=0.485 initial=0.016 final=0.600 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.062 net=0.051 yaw_rate_ratio=0.324
  g_v0p22_fwd_w0p3: err rms=0.557 initial=0.016 final=0.600 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.063 net=0.050 yaw_rate_ratio=0.337
  g_v0p22_lat_w0: err rms=0.021 initial=0.017 final=0.019 max=0.049 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.002 yaw_rate_ratio=n/a
  g_v0p22_lat_w0p08: err rms=0.156 initial=0.015 final=0.152 max=0.238 closed=n/a | settle=11.750 (n=46/100) | progress=0.078 net=0.012 yaw_rate_ratio=0.534
  g_v0p22_lat_w0p15: err rms=0.255 initial=0.016 final=0.277 max=0.361 closed=n/a | settle=n/a (n=0/100) | progress=0.138 net=0.022 yaw_rate_ratio=0.562
  g_v0p22_lat_w0p3: err rms=0.539 initial=0.013 final=0.600 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.169 net=0.050 yaw_rate_ratio=0.628
  g_v0p3_fwd_w0: err rms=0.045 initial=0.015 final=0.074 max=0.100 closed=n/a | settle=0.000 (n=100/100) | progress=-0.003 net=0.004 yaw_rate_ratio=0.359
  g_v0p3_fwd_w0p08: err rms=0.392 initial=0.015 final=0.596 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.033 net=0.050 yaw_rate_ratio=0.358
  g_v0p3_fwd_w0p15: err rms=0.511 initial=0.013 final=0.600 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.042 net=0.050 yaw_rate_ratio=0.354
  g_v0p3_fwd_w0p3: err rms=0.562 initial=0.015 final=0.600 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.043 net=0.050 yaw_rate_ratio=0.351
  g_v0p3_lat_w0: err rms=0.018 initial=0.015 final=0.014 max=0.042 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
  g_v0p3_lat_w0p08: err rms=0.157 initial=0.014 final=0.166 max=0.233 closed=n/a | settle=11.820 (n=33/100) | progress=0.076 net=0.013 yaw_rate_ratio=0.533
  g_v0p3_lat_w0p15: err rms=0.306 initial=0.016 final=0.397 max=0.431 closed=n/a | settle=n/a (n=0/100) | progress=0.126 net=0.032 yaw_rate_ratio=0.535
  g_v0p3_lat_w0p3: err rms=0.544 initial=0.015 final=0.600 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.153 net=0.049 yaw_rate_ratio=0.580

body yaw by hold (window [4 s, hold_s); medians; net = net yaw rate rad/s, ratio = net / wz_cmd SIGNED (the gated turn number), wz_ratio = mean wz / wz_cmd; ripple = std of the 1-s mean wz; drift = |mean wz| on wz = 0 holds; planar = xy drift m/s on pure turns; body = along-bearing speed / |v_cmd|, cross = mean |perp| m/s):
          settle: net=n/a ratio=n/a wz_ratio=n/a ripple=n/a | drift=n/a planar=n/a | body=n/a cross=n/a (n=0)
         g_v0_w0: net=0.002 ratio=n/a wz_ratio=n/a ripple=0.016 | drift=0.003 planar=n/a | body=n/a cross=n/a (n=100)
      g_v0_w0p08: net=0.077 ratio=0.956 wz_ratio=0.961 ripple=0.038 | drift=n/a planar=0.010 | body=n/a cross=n/a (n=100)
      g_v0_w0p15: net=0.144 ratio=0.960 wz_ratio=0.965 ripple=0.033 | drift=n/a planar=0.009 | body=n/a cross=n/a (n=100)
       g_v0_w0p3: net=0.184 ratio=0.614 wz_ratio=0.616 ripple=0.026 | drift=n/a planar=0.017 | body=n/a cross=n/a (n=100)
  g_v0p22_fwd_w0: net=-0.000 ratio=n/a wz_ratio=n/a ripple=0.057 | drift=0.004 planar=n/a | body=1.007 cross=0.026 (n=100)
  g_v0p22_fwd_w0p08: net=0.051 ratio=0.642 wz_ratio=0.639 ripple=0.033 | drift=n/a planar=n/a | body=0.833 cross=0.027 (n=100)
  g_v0p22_fwd_w0p15: net=0.058 ratio=0.387 wz_ratio=0.380 ripple=0.037 | drift=n/a planar=n/a | body=0.703 cross=0.028 (n=100)
  g_v0p22_fwd_w0p3: net=0.057 ratio=0.188 wz_ratio=0.184 ripple=0.025 | drift=n/a planar=n/a | body=0.674 cross=0.027 (n=100)
  g_v0p22_lat_w0: net=-0.002 ratio=n/a wz_ratio=n/a ripple=0.015 | drift=0.003 planar=n/a | body=1.046 cross=0.026 (n=100)
  g_v0p22_lat_w0p08: net=0.084 ratio=1.055 wz_ratio=1.061 ripple=0.030 | drift=n/a planar=n/a | body=0.983 cross=0.040 (n=100)
  g_v0p22_lat_w0p15: net=0.153 ratio=1.019 wz_ratio=1.025 ripple=0.024 | drift=n/a planar=n/a | body=0.953 cross=0.040 (n=100)
  g_v0p22_lat_w0p3: net=0.179 ratio=0.595 wz_ratio=0.597 ripple=0.029 | drift=n/a planar=n/a | body=0.879 cross=0.051 (n=100)
   g_v0p3_fwd_w0: net=-0.003 ratio=n/a wz_ratio=n/a ripple=0.025 | drift=0.004 planar=n/a | body=0.911 cross=0.026 (n=100)
  g_v0p3_fwd_w0p08: net=0.036 ratio=0.455 wz_ratio=0.449 ripple=0.035 | drift=n/a planar=n/a | body=0.744 cross=0.031 (n=100)
  g_v0p3_fwd_w0p15: net=0.045 ratio=0.302 wz_ratio=0.294 ripple=0.029 | drift=n/a planar=n/a | body=0.679 cross=0.034 (n=100)
  g_v0p3_fwd_w0p3: net=0.044 ratio=0.145 wz_ratio=0.143 ripple=0.031 | drift=n/a planar=n/a | body=0.659 cross=0.034 (n=100)
   g_v0p3_lat_w0: net=-0.000 ratio=n/a wz_ratio=n/a ripple=0.014 | drift=0.002 planar=n/a | body=0.926 cross=0.030 (n=100)
  g_v0p3_lat_w0p08: net=0.082 ratio=1.022 wz_ratio=1.024 ripple=0.026 | drift=n/a planar=n/a | body=0.878 cross=0.040 (n=100)
  g_v0p3_lat_w0p15: net=0.140 ratio=0.931 wz_ratio=0.934 ripple=0.026 | drift=n/a planar=n/a | body=0.826 cross=0.040 (n=100)
  g_v0p3_lat_w0p3: net=0.163 ratio=0.545 wz_ratio=0.546 ripple=0.033 | drift=n/a planar=n/a | body=0.766 cross=0.055 (n=100)

by category (20 waves, 2000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: n/a):
    g_v0_w0: n=100 surv=1.000 (g_v0_w0=1.00) | ratio=n/a achieved=0.003 tripod=0.762 | closed=n/a progress=n/a net=0.000 | drift=0.035 dir_err=n/a
  g_v0_w0p08: n=100 surv=1.000 (g_v0_w0p08=1.00) | ratio=n/a achieved=-0.002 tripod=0.760 | closed=n/a progress=0.076 net=0.013 | drift=0.041 dir_err=n/a
  g_v0_w0p15: n=100 surv=1.000 (g_v0_w0p15=1.00) | ratio=n/a achieved=-0.003 tripod=0.770 | closed=n/a progress=0.140 net=0.019 | drift=0.043 dir_err=n/a
  g_v0_w0p3: n=100 surv=1.000 (g_v0_w0p3=1.00) | ratio=n/a achieved=-0.010 tripod=0.778 | closed=n/a progress=0.180 net=0.049 | drift=0.052 dir_err=n/a
  g_v0p22_fwd_w0: n=100 surv=1.000 (g_v0p22_fwd_w0=1.00) | ratio=1.014 achieved=0.223 tripod=0.690 | closed=n/a progress=-0.001 net=0.001 | drift=0.224 dir_err=0.030
  g_v0p22_fwd_w0p08: n=100 surv=1.000 (g_v0p22_fwd_w0p08=1.00) | ratio=0.877 achieved=0.193 tripod=0.694 | closed=n/a progress=0.052 net=0.031 | drift=0.196 dir_err=0.016
  g_v0p22_fwd_w0p15: n=100 surv=1.000 (g_v0p22_fwd_w0p15=1.00) | ratio=0.761 achieved=0.167 tripod=0.662 | closed=n/a progress=0.062 net=0.051 | drift=0.174 dir_err=0.029
  g_v0p22_fwd_w0p3: n=100 surv=1.000 (g_v0p22_fwd_w0p3=1.00) | ratio=0.715 achieved=0.157 tripod=0.651 | closed=n/a progress=0.063 net=0.050 | drift=0.165 dir_err=0.041
  g_v0p22_lat_w0: n=100 surv=1.000 (g_v0p22_lat_w0=1.00) | ratio=1.044 achieved=0.230 tripod=0.664 | closed=n/a progress=n/a net=0.002 | drift=0.232 dir_err=0.007
  g_v0p22_lat_w0p08: n=100 surv=1.000 (g_v0p22_lat_w0p08=1.00) | ratio=0.997 achieved=0.219 tripod=0.661 | closed=n/a progress=0.078 net=0.012 | drift=0.225 dir_err=0.041
  g_v0p22_lat_w0p15: n=100 surv=1.000 (g_v0p22_lat_w0p15=1.00) | ratio=0.964 achieved=0.212 tripod=0.682 | closed=n/a progress=0.138 net=0.022 | drift=0.220 dir_err=0.032
  g_v0p22_lat_w0p3: n=100 surv=1.000 (g_v0p22_lat_w0p3=1.00) | ratio=0.891 achieved=0.196 tripod=0.757 | closed=n/a progress=0.169 net=0.050 | drift=0.209 dir_err=0.016
  g_v0p3_fwd_w0: n=100 surv=1.000 (g_v0p3_fwd_w0=1.00) | ratio=0.922 achieved=0.277 tripod=0.594 | closed=n/a progress=-0.003 net=0.004 | drift=0.277 dir_err=0.014
  g_v0p3_fwd_w0p08: n=100 surv=1.000 (g_v0p3_fwd_w0p08=1.00) | ratio=0.783 achieved=0.235 tripod=0.462 | closed=n/a progress=0.033 net=0.050 | drift=0.241 dir_err=0.038
  g_v0p3_fwd_w0p15: n=100 surv=1.000 (g_v0p3_fwd_w0p15=1.00) | ratio=0.726 achieved=0.218 tripod=0.450 | closed=n/a progress=0.042 net=0.050 | drift=0.226 dir_err=0.053
  g_v0p3_fwd_w0p3: n=100 surv=1.000 (g_v0p3_fwd_w0p3=1.00) | ratio=0.690 achieved=0.207 tripod=0.423 | closed=n/a progress=0.043 net=0.050 | drift=0.216 dir_err=0.051
  g_v0p3_lat_w0: n=100 surv=1.000 (g_v0p3_lat_w0=1.00) | ratio=0.925 achieved=0.277 tripod=0.674 | closed=n/a progress=n/a net=0.001 | drift=0.279 dir_err=0.011
  g_v0p3_lat_w0p08: n=100 surv=1.000 (g_v0p3_lat_w0p08=1.00) | ratio=0.884 achieved=0.265 tripod=0.685 | closed=n/a progress=0.076 net=0.013 | drift=0.271 dir_err=0.047
  g_v0p3_lat_w0p15: n=100 surv=1.000 (g_v0p3_lat_w0p15=1.00) | ratio=0.844 achieved=0.253 tripod=0.718 | closed=n/a progress=0.126 net=0.032 | drift=0.259 dir_err=0.042
  g_v0p3_lat_w0p3: n=100 surv=1.000 (g_v0p3_lat_w0p3=1.00) | ratio=0.784 achieved=0.235 tripod=0.737 | closed=n/a progress=0.153 net=0.049 | drift=0.249 dir_err=0.029

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
  g_v0_w0p08: net_yaw_rate=0.077 net_yaw_ratio=0.956 wz_ratio=0.961 yaw_ripple=0.038 heading_drift=n/a planar_drift=0.010 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  g_v0_w0p15: net_yaw_rate=0.144 net_yaw_ratio=0.960 wz_ratio=0.965 yaw_ripple=0.033 heading_drift=n/a planar_drift=0.009 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  g_v0_w0p3: net_yaw_rate=0.184 net_yaw_ratio=0.614 wz_ratio=0.616 yaw_ripple=0.026 heading_drift=n/a planar_drift=0.017 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  g_v0p22_fwd_w0: net_yaw_rate=-0.000 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.057 heading_drift=0.004 planar_drift=n/a body_ratio=1.007 cross_body=0.026 adapter_clipped_frac=n/a
  g_v0p22_fwd_w0p08: net_yaw_rate=0.051 net_yaw_ratio=0.642 wz_ratio=0.639 yaw_ripple=0.033 heading_drift=n/a planar_drift=n/a body_ratio=0.833 cross_body=0.027 adapter_clipped_frac=n/a
  g_v0p22_fwd_w0p15: net_yaw_rate=0.058 net_yaw_ratio=0.387 wz_ratio=0.380 yaw_ripple=0.037 heading_drift=n/a planar_drift=n/a body_ratio=0.703 cross_body=0.028 adapter_clipped_frac=n/a
  g_v0p22_fwd_w0p3: net_yaw_rate=0.057 net_yaw_ratio=0.188 wz_ratio=0.184 yaw_ripple=0.025 heading_drift=n/a planar_drift=n/a body_ratio=0.674 cross_body=0.027 adapter_clipped_frac=n/a
  g_v0p22_lat_w0: net_yaw_rate=-0.002 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.015 heading_drift=0.003 planar_drift=n/a body_ratio=1.046 cross_body=0.026 adapter_clipped_frac=n/a
  g_v0p22_lat_w0p08: net_yaw_rate=0.084 net_yaw_ratio=1.055 wz_ratio=1.061 yaw_ripple=0.030 heading_drift=n/a planar_drift=n/a body_ratio=0.983 cross_body=0.040 adapter_clipped_frac=n/a
  g_v0p22_lat_w0p15: net_yaw_rate=0.153 net_yaw_ratio=1.019 wz_ratio=1.025 yaw_ripple=0.024 heading_drift=n/a planar_drift=n/a body_ratio=0.953 cross_body=0.040 adapter_clipped_frac=n/a
  g_v0p22_lat_w0p3: net_yaw_rate=0.179 net_yaw_ratio=0.595 wz_ratio=0.597 yaw_ripple=0.029 heading_drift=n/a planar_drift=n/a body_ratio=0.879 cross_body=0.051 adapter_clipped_frac=n/a
  g_v0p3_fwd_w0: net_yaw_rate=-0.003 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.025 heading_drift=0.004 planar_drift=n/a body_ratio=0.911 cross_body=0.026 adapter_clipped_frac=n/a
  g_v0p3_fwd_w0p08: net_yaw_rate=0.036 net_yaw_ratio=0.455 wz_ratio=0.449 yaw_ripple=0.035 heading_drift=n/a planar_drift=n/a body_ratio=0.744 cross_body=0.031 adapter_clipped_frac=n/a
  g_v0p3_fwd_w0p15: net_yaw_rate=0.045 net_yaw_ratio=0.302 wz_ratio=0.294 yaw_ripple=0.029 heading_drift=n/a planar_drift=n/a body_ratio=0.679 cross_body=0.034 adapter_clipped_frac=n/a
  g_v0p3_fwd_w0p3: net_yaw_rate=0.044 net_yaw_ratio=0.145 wz_ratio=0.143 yaw_ripple=0.031 heading_drift=n/a planar_drift=n/a body_ratio=0.659 cross_body=0.034 adapter_clipped_frac=n/a
  g_v0p3_lat_w0: net_yaw_rate=-0.000 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.014 heading_drift=0.002 planar_drift=n/a body_ratio=0.926 cross_body=0.030 adapter_clipped_frac=n/a
  g_v0p3_lat_w0p08: net_yaw_rate=0.082 net_yaw_ratio=1.022 wz_ratio=1.024 yaw_ripple=0.026 heading_drift=n/a planar_drift=n/a body_ratio=0.878 cross_body=0.040 adapter_clipped_frac=n/a
  g_v0p3_lat_w0p15: net_yaw_rate=0.140 net_yaw_ratio=0.931 wz_ratio=0.934 yaw_ripple=0.026 heading_drift=n/a planar_drift=n/a body_ratio=0.826 cross_body=0.040 adapter_clipped_frac=n/a
  g_v0p3_lat_w0p3: net_yaw_rate=0.163 net_yaw_ratio=0.545 wz_ratio=0.546 yaw_ripple=0.033 heading_drift=n/a planar_drift=n/a body_ratio=0.766 cross_body=0.055 adapter_clipped_frac=n/a

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
