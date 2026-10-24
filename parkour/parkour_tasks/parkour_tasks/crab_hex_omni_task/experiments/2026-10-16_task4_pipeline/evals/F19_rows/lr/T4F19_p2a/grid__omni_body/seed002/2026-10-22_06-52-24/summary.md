```
=== crab-hex gait eval ===
scenario   : grid__omni_body  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-21_23-06-12_seed19_X06_p2a_A15pB/model_19994.pt
episodes   : 2000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6912765087365345  p25=0.6057827837031613  p75=0.7400364435330131
tippy_tap_fraction median=0.44155844155844154
slip_ratio         median=0.2281590447053626
tracking_ratio     median=0.8832529530776463  (achieved/commanded vx, walking holds; n=1600)
schedule_completion_rate=0.979
terminations={'schedule_complete': 1958, 'fall': 42}
facing_error_rms   median=0.2455870764600361  (rad, whole episode; n=2000)

by hold:
    settle: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.7802881732990194 (n=1968)
   g_v0_w0: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.7961421512757512 (n=100)
  g_v0_w0p08: cmd 0.00 -> achieved -0.005 m/s | tripod median=0.7492077196537614 (n=100)
  g_v0_w0p15: cmd 0.00 -> achieved -0.001 m/s | tripod median=0.7807141337326149 (n=100)
  g_v0_w0p3: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.801984801691001 (n=100)
  g_v0p22_fwd_w0: cmd 0.22 -> achieved 0.236 m/s | tripod median=0.7140064408922804 (n=100)
  g_v0p22_fwd_w0p08: cmd 0.22 -> achieved 0.199 m/s | tripod median=0.618049844593491 (n=100)
  g_v0p22_fwd_w0p15: cmd 0.22 -> achieved 0.179 m/s | tripod median=0.6132936446646692 (n=100)
  g_v0p22_fwd_w0p3: cmd 0.22 -> achieved 0.169 m/s | tripod median=0.5995706393249973 (n=100)
  g_v0p22_lat_w0: cmd 0.22 -> achieved 0.233 m/s | tripod median=0.7267675367032891 (n=100)
  g_v0p22_lat_w0p08: cmd 0.22 -> achieved 0.225 m/s | tripod median=0.6682487505531518 (n=100)
  g_v0p22_lat_w0p15: cmd 0.22 -> achieved 0.213 m/s | tripod median=0.7056628398661573 (n=100)
  g_v0p22_lat_w0p3: cmd 0.22 -> achieved 0.195 m/s | tripod median=0.734324256965251 (n=100)
  g_v0p3_fwd_w0: cmd 0.30 -> achieved 0.275 m/s | tripod median=0.4377046511737812 (n=100)
  g_v0p3_fwd_w0p08: cmd 0.30 -> achieved 0.214 m/s | tripod median=0.3788108925245921 (n=100)
  g_v0p3_fwd_w0p15: cmd 0.30 -> achieved 0.200 m/s | tripod median=0.3701578014418554 (n=100)
  g_v0p3_fwd_w0p3: cmd 0.30 -> achieved 0.184 m/s | tripod median=0.34911632326522274 (n=100)
  g_v0p3_lat_w0: cmd 0.30 -> achieved 0.277 m/s | tripod median=0.6889963594309494 (n=100)
  g_v0p3_lat_w0p08: cmd 0.30 -> achieved 0.270 m/s | tripod median=0.6602903663535442 (n=100)
  g_v0p3_lat_w0p15: cmd 0.30 -> achieved 0.255 m/s | tripod median=0.6986646100749634 (n=100)
  g_v0p3_lat_w0p3: cmd 0.30 -> achieved 0.235 m/s | tripod median=0.7199483264815649 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.018 initial=0.006 final=0.015 max=0.034 closed=n/a | settle=0.000 (n=2000/2000) | progress=n/a net=0.001 yaw_rate_ratio=n/a
   g_v0_w0: err rms=0.029 initial=0.013 final=0.028 max=0.062 closed=n/a | settle=0.000 (n=100/100) | progress=0.003 net=0.001 yaw_rate_ratio=0.279
  g_v0_w0p08: err rms=0.150 initial=0.017 final=0.165 max=0.233 closed=n/a | settle=11.840 (n=33/100) | progress=0.077 net=0.013 yaw_rate_ratio=0.591
  g_v0_w0p15: err rms=0.271 initial=0.014 final=0.350 max=0.400 closed=n/a | settle=n/a (n=0/100) | progress=0.131 net=0.026 yaw_rate_ratio=0.583
  g_v0_w0p3: err rms=0.539 initial=0.014 final=0.600 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.158 net=0.050 yaw_rate_ratio=0.581
  g_v0p22_fwd_w0: err rms=0.067 initial=0.014 final=0.099 max=0.122 closed=n/a | settle=0.000 (n=100/100) | progress=-0.001 net=-0.004 yaw_rate_ratio=0.309
  g_v0p22_fwd_w0p08: err rms=0.275 initial=0.015 final=0.437 max=0.441 closed=n/a | settle=n/a (n=0/100) | progress=0.049 net=0.035 yaw_rate_ratio=0.332
  g_v0p22_fwd_w0p15: err rms=0.483 initial=0.017 final=0.600 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.064 net=0.050 yaw_rate_ratio=0.336
  g_v0p22_fwd_w0p3: err rms=0.559 initial=0.016 final=0.600 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.065 net=0.050 yaw_rate_ratio=0.352
  g_v0p22_lat_w0: err rms=0.026 initial=0.013 final=0.027 max=0.055 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.000 yaw_rate_ratio=n/a
  g_v0p22_lat_w0p08: err rms=0.154 initial=0.015 final=0.166 max=0.231 closed=n/a | settle=11.800 (n=29/100) | progress=0.077 net=0.012 yaw_rate_ratio=0.532
  g_v0p22_lat_w0p15: err rms=0.265 initial=0.014 final=0.289 max=0.362 closed=n/a | settle=n/a (n=0/100) | progress=0.136 net=0.023 yaw_rate_ratio=0.539
  g_v0p22_lat_w0p3: err rms=0.546 initial=0.014 final=0.600 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.147 net=0.050 yaw_rate_ratio=0.559
  g_v0p3_fwd_w0: err rms=0.046 initial=0.016 final=0.065 max=0.107 closed=n/a | settle=0.000 (n=97/100) | progress=-0.003 net=0.004 yaw_rate_ratio=0.663
  g_v0p3_fwd_w0p08: err rms=0.354 initial=0.016 final=0.514 max=0.533 closed=n/a | settle=n/a (n=0/100) | progress=0.041 net=0.042 yaw_rate_ratio=0.406
  g_v0p3_fwd_w0p15: err rms=0.508 initial=0.012 final=0.600 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.051 net=0.050 yaw_rate_ratio=0.388
  g_v0p3_fwd_w0p3: err rms=0.563 initial=0.010 final=0.600 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.054 net=0.050 yaw_rate_ratio=0.381
  g_v0p3_lat_w0: err rms=0.031 initial=0.016 final=0.032 max=0.060 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.000 yaw_rate_ratio=n/a
  g_v0p3_lat_w0p08: err rms=0.170 initial=0.012 final=0.196 max=0.249 closed=n/a | settle=11.760 (n=13/100) | progress=0.073 net=0.014 yaw_rate_ratio=0.466
  g_v0p3_lat_w0p15: err rms=0.350 initial=0.012 final=0.443 max=0.490 closed=n/a | settle=n/a (n=0/100) | progress=0.121 net=0.035 yaw_rate_ratio=0.500
  g_v0p3_lat_w0p3: err rms=0.549 initial=0.015 final=0.600 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.139 net=0.050 yaw_rate_ratio=0.527

body yaw by hold (window [4 s, hold_s); medians; net = net yaw rate rad/s, ratio = net / wz_cmd SIGNED (the gated turn number), wz_ratio = mean wz / wz_cmd; ripple = std of the 1-s mean wz; drift = |mean wz| on wz = 0 holds; planar = xy drift m/s on pure turns; body = along-bearing speed / |v_cmd|, cross = mean |perp| m/s):
          settle: net=n/a ratio=n/a wz_ratio=n/a ripple=n/a | drift=n/a planar=n/a | body=n/a cross=n/a (n=0)
         g_v0_w0: net=0.001 ratio=n/a wz_ratio=n/a ripple=0.018 | drift=0.002 planar=n/a | body=n/a cross=n/a (n=100)
      g_v0_w0p08: net=0.079 ratio=0.990 wz_ratio=1.001 ripple=0.029 | drift=n/a planar=0.007 | body=n/a cross=n/a (n=100)
      g_v0_w0p15: net=0.135 ratio=0.903 wz_ratio=0.909 ripple=0.029 | drift=n/a planar=0.006 | body=n/a cross=n/a (n=100)
       g_v0_w0p3: net=0.155 ratio=0.518 wz_ratio=0.521 ripple=0.026 | drift=n/a planar=0.009 | body=n/a cross=n/a (n=100)
  g_v0p22_fwd_w0: net=0.007 ratio=n/a wz_ratio=n/a ripple=0.022 | drift=0.008 planar=n/a | body=1.073 cross=0.025 (n=100)
  g_v0p22_fwd_w0p08: net=0.052 ratio=0.648 wz_ratio=0.647 ripple=0.027 | drift=n/a planar=n/a | body=0.859 cross=0.030 (n=100)
  g_v0p22_fwd_w0p15: net=0.065 ratio=0.432 wz_ratio=0.430 ripple=0.029 | drift=n/a planar=n/a | body=0.765 cross=0.029 (n=100)
  g_v0p22_fwd_w0p3: net=0.063 ratio=0.209 wz_ratio=0.206 ripple=0.026 | drift=n/a planar=n/a | body=0.743 cross=0.030 (n=100)
  g_v0p22_lat_w0: net=0.001 ratio=n/a wz_ratio=n/a ripple=0.012 | drift=0.002 planar=n/a | body=1.065 cross=0.022 (n=100)
  g_v0p22_lat_w0p08: net=0.079 ratio=0.984 wz_ratio=0.991 ripple=0.029 | drift=n/a planar=n/a | body=1.012 cross=0.039 (n=100)
  g_v0p22_lat_w0p15: net=0.151 ratio=1.005 wz_ratio=1.006 ripple=0.025 | drift=n/a planar=n/a | body=0.945 cross=0.044 (n=100)
  g_v0p22_lat_w0p3: net=0.150 ratio=0.500 wz_ratio=0.501 ripple=0.020 | drift=n/a planar=n/a | body=0.863 cross=0.055 (n=100)
   g_v0p3_fwd_w0: net=-0.004 ratio=n/a wz_ratio=n/a ripple=0.039 | drift=0.004 planar=n/a | body=0.889 cross=0.031 (n=100)
  g_v0p3_fwd_w0p08: net=0.050 ratio=0.630 wz_ratio=0.607 ripple=0.035 | drift=n/a planar=n/a | body=0.620 cross=0.039 (n=100)
  g_v0p3_fwd_w0p15: net=0.061 ratio=0.404 wz_ratio=0.390 ripple=0.029 | drift=n/a planar=n/a | body=0.584 cross=0.038 (n=100)
  g_v0p3_fwd_w0p3: net=0.061 ratio=0.202 wz_ratio=0.192 ripple=0.030 | drift=n/a planar=n/a | body=0.538 cross=0.038 (n=100)
   g_v0p3_lat_w0: net=0.000 ratio=n/a wz_ratio=n/a ripple=0.015 | drift=0.002 planar=n/a | body=0.927 cross=0.030 (n=100)
  g_v0p3_lat_w0p08: net=0.076 ratio=0.951 wz_ratio=0.957 ripple=0.025 | drift=n/a planar=n/a | body=0.893 cross=0.043 (n=100)
  g_v0p3_lat_w0p15: net=0.134 ratio=0.895 wz_ratio=0.898 ripple=0.025 | drift=n/a planar=n/a | body=0.833 cross=0.047 (n=100)
  g_v0p3_lat_w0p3: net=0.147 ratio=0.489 wz_ratio=0.490 ripple=0.024 | drift=n/a planar=n/a | body=0.755 cross=0.058 (n=100)

by category (20 waves, 2000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: n/a):
    g_v0_w0: n=100 surv=1.000 (g_v0_w0=1.00) | ratio=n/a achieved=0.002 tripod=0.796 | closed=n/a progress=0.003 net=0.001 | drift=0.038 dir_err=n/a
  g_v0_w0p08: n=100 surv=1.000 (g_v0_w0p08=1.00) | ratio=n/a achieved=-0.005 tripod=0.749 | closed=n/a progress=0.077 net=0.013 | drift=0.042 dir_err=n/a
  g_v0_w0p15: n=100 surv=1.000 (g_v0_w0p15=1.00) | ratio=n/a achieved=-0.001 tripod=0.781 | closed=n/a progress=0.131 net=0.026 | drift=0.044 dir_err=n/a
  g_v0_w0p3: n=100 surv=1.000 (g_v0_w0p3=1.00) | ratio=n/a achieved=0.001 tripod=0.802 | closed=n/a progress=0.158 net=0.050 | drift=0.050 dir_err=n/a
  g_v0p22_fwd_w0: n=100 surv=1.000 (g_v0p22_fwd_w0=1.00) | ratio=1.075 achieved=0.236 tripod=0.714 | closed=n/a progress=-0.001 net=-0.004 | drift=0.234 dir_err=0.063
  g_v0p22_fwd_w0p08: n=100 surv=1.000 (g_v0p22_fwd_w0p08=1.00) | ratio=0.904 achieved=0.199 tripod=0.618 | closed=n/a progress=0.049 net=0.035 | drift=0.201 dir_err=0.071
  g_v0p22_fwd_w0p15: n=100 surv=1.000 (g_v0p22_fwd_w0p15=1.00) | ratio=0.813 achieved=0.179 tripod=0.613 | closed=n/a progress=0.064 net=0.050 | drift=0.183 dir_err=0.070
  g_v0p22_fwd_w0p3: n=100 surv=1.000 (g_v0p22_fwd_w0p3=1.00) | ratio=0.769 achieved=0.169 tripod=0.600 | closed=n/a progress=0.065 net=0.050 | drift=0.176 dir_err=0.071
  g_v0p22_lat_w0: n=100 surv=1.000 (g_v0p22_lat_w0=1.00) | ratio=1.060 achieved=0.233 tripod=0.727 | closed=n/a progress=n/a net=-0.000 | drift=0.233 dir_err=0.012
  g_v0p22_lat_w0p08: n=100 surv=1.000 (g_v0p22_lat_w0p08=1.00) | ratio=1.022 achieved=0.225 tripod=0.668 | closed=n/a progress=0.077 net=0.012 | drift=0.228 dir_err=0.032
  g_v0p22_lat_w0p15: n=100 surv=1.000 (g_v0p22_lat_w0p15=1.00) | ratio=0.968 achieved=0.213 tripod=0.706 | closed=n/a progress=0.136 net=0.023 | drift=0.220 dir_err=0.046
  g_v0p22_lat_w0p3: n=100 surv=1.000 (g_v0p22_lat_w0p3=1.00) | ratio=0.888 achieved=0.195 tripod=0.734 | closed=n/a progress=0.147 net=0.050 | drift=0.209 dir_err=0.133
  g_v0p3_fwd_w0: n=100 surv=1.000 (g_v0p3_fwd_w0=1.00) | ratio=0.916 achieved=0.275 tripod=0.438 | closed=n/a progress=-0.003 net=0.004 | drift=0.276 dir_err=0.037
  g_v0p3_fwd_w0p08: n=100 surv=0.850 (g_v0p3_fwd_w0p08=0.85) | ratio=0.714 achieved=0.214 tripod=0.379 | closed=n/a progress=0.041 net=0.042 | drift=0.229 dir_err=0.067
  g_v0p3_fwd_w0p15: n=100 surv=0.840 (g_v0p3_fwd_w0p15=0.84) | ratio=0.667 achieved=0.200 tripod=0.370 | closed=n/a progress=0.051 net=0.050 | drift=0.216 dir_err=0.071
  g_v0p3_fwd_w0p3: n=100 surv=0.890 (g_v0p3_fwd_w0p3=0.89) | ratio=0.612 achieved=0.184 tripod=0.349 | closed=n/a progress=0.054 net=0.050 | drift=0.207 dir_err=0.053
  g_v0p3_lat_w0: n=100 surv=1.000 (g_v0p3_lat_w0=1.00) | ratio=0.923 achieved=0.277 tripod=0.689 | closed=n/a progress=n/a net=-0.000 | drift=0.277 dir_err=0.018
  g_v0p3_lat_w0p08: n=100 surv=1.000 (g_v0p3_lat_w0p08=1.00) | ratio=0.900 achieved=0.270 tripod=0.660 | closed=n/a progress=0.073 net=0.014 | drift=0.274 dir_err=0.042
  g_v0p3_lat_w0p15: n=100 surv=1.000 (g_v0p3_lat_w0p15=1.00) | ratio=0.851 achieved=0.255 tripod=0.699 | closed=n/a progress=0.121 net=0.035 | drift=0.261 dir_err=0.081
  g_v0p3_lat_w0p3: n=100 surv=1.000 (g_v0p3_lat_w0p3=1.00) | ratio=0.784 achieved=0.235 tripod=0.720 | closed=n/a progress=0.139 net=0.050 | drift=0.248 dir_err=0.128

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
    g_v0_w0: net_yaw_rate=0.001 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.018 heading_drift=0.002 planar_drift=n/a body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  g_v0_w0p08: net_yaw_rate=0.079 net_yaw_ratio=0.990 wz_ratio=1.001 yaw_ripple=0.029 heading_drift=n/a planar_drift=0.007 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  g_v0_w0p15: net_yaw_rate=0.135 net_yaw_ratio=0.903 wz_ratio=0.909 yaw_ripple=0.029 heading_drift=n/a planar_drift=0.006 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  g_v0_w0p3: net_yaw_rate=0.155 net_yaw_ratio=0.518 wz_ratio=0.521 yaw_ripple=0.026 heading_drift=n/a planar_drift=0.009 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  g_v0p22_fwd_w0: net_yaw_rate=0.007 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.022 heading_drift=0.008 planar_drift=n/a body_ratio=1.073 cross_body=0.025 adapter_clipped_frac=n/a
  g_v0p22_fwd_w0p08: net_yaw_rate=0.052 net_yaw_ratio=0.648 wz_ratio=0.647 yaw_ripple=0.027 heading_drift=n/a planar_drift=n/a body_ratio=0.859 cross_body=0.030 adapter_clipped_frac=n/a
  g_v0p22_fwd_w0p15: net_yaw_rate=0.065 net_yaw_ratio=0.432 wz_ratio=0.430 yaw_ripple=0.029 heading_drift=n/a planar_drift=n/a body_ratio=0.765 cross_body=0.029 adapter_clipped_frac=n/a
  g_v0p22_fwd_w0p3: net_yaw_rate=0.063 net_yaw_ratio=0.209 wz_ratio=0.206 yaw_ripple=0.026 heading_drift=n/a planar_drift=n/a body_ratio=0.743 cross_body=0.030 adapter_clipped_frac=n/a
  g_v0p22_lat_w0: net_yaw_rate=0.001 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.012 heading_drift=0.002 planar_drift=n/a body_ratio=1.065 cross_body=0.022 adapter_clipped_frac=n/a
  g_v0p22_lat_w0p08: net_yaw_rate=0.079 net_yaw_ratio=0.984 wz_ratio=0.991 yaw_ripple=0.029 heading_drift=n/a planar_drift=n/a body_ratio=1.012 cross_body=0.039 adapter_clipped_frac=n/a
  g_v0p22_lat_w0p15: net_yaw_rate=0.151 net_yaw_ratio=1.005 wz_ratio=1.006 yaw_ripple=0.025 heading_drift=n/a planar_drift=n/a body_ratio=0.945 cross_body=0.044 adapter_clipped_frac=n/a
  g_v0p22_lat_w0p3: net_yaw_rate=0.150 net_yaw_ratio=0.500 wz_ratio=0.501 yaw_ripple=0.020 heading_drift=n/a planar_drift=n/a body_ratio=0.863 cross_body=0.055 adapter_clipped_frac=n/a
  g_v0p3_fwd_w0: net_yaw_rate=-0.004 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.039 heading_drift=0.004 planar_drift=n/a body_ratio=0.889 cross_body=0.031 adapter_clipped_frac=n/a
  g_v0p3_fwd_w0p08: net_yaw_rate=0.050 net_yaw_ratio=0.630 wz_ratio=0.607 yaw_ripple=0.035 heading_drift=n/a planar_drift=n/a body_ratio=0.620 cross_body=0.039 adapter_clipped_frac=n/a
  g_v0p3_fwd_w0p15: net_yaw_rate=0.061 net_yaw_ratio=0.404 wz_ratio=0.390 yaw_ripple=0.029 heading_drift=n/a planar_drift=n/a body_ratio=0.584 cross_body=0.038 adapter_clipped_frac=n/a
  g_v0p3_fwd_w0p3: net_yaw_rate=0.061 net_yaw_ratio=0.202 wz_ratio=0.192 yaw_ripple=0.030 heading_drift=n/a planar_drift=n/a body_ratio=0.538 cross_body=0.038 adapter_clipped_frac=n/a
  g_v0p3_lat_w0: net_yaw_rate=0.000 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.015 heading_drift=0.002 planar_drift=n/a body_ratio=0.927 cross_body=0.030 adapter_clipped_frac=n/a
  g_v0p3_lat_w0p08: net_yaw_rate=0.076 net_yaw_ratio=0.951 wz_ratio=0.957 yaw_ripple=0.025 heading_drift=n/a planar_drift=n/a body_ratio=0.893 cross_body=0.043 adapter_clipped_frac=n/a
  g_v0p3_lat_w0p15: net_yaw_rate=0.134 net_yaw_ratio=0.895 wz_ratio=0.898 yaw_ripple=0.025 heading_drift=n/a planar_drift=n/a body_ratio=0.833 cross_body=0.047 adapter_clipped_frac=n/a
  g_v0p3_lat_w0p3: net_yaw_rate=0.147 net_yaw_ratio=0.489 wz_ratio=0.490 yaw_ripple=0.024 heading_drift=n/a planar_drift=n/a body_ratio=0.755 cross_body=0.058 adapter_clipped_frac=n/a

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
