```
=== crab-hex gait eval ===
scenario   : grid__omni_body  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-21_23-06-12_seed19_X06_p2a_A15pB/model_19994.pt
episodes   : 2000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6892825957095781  p25=0.5996898914251334  p75=0.7426513223561122
tippy_tap_fraction median=0.4375
slip_ratio         median=0.2264402208653817
tracking_ratio     median=0.8827773230992417  (achieved/commanded vx, walking holds; n=1600)
schedule_completion_rate=0.9815
terminations={'schedule_complete': 1963, 'fall': 37}
facing_error_rms   median=0.24971359285497552  (rad, whole episode; n=2000)

by hold:
    settle: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7758775097038262 (n=1966)
   g_v0_w0: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.7997880648898692 (n=100)
  g_v0_w0p08: cmd 0.00 -> achieved -0.004 m/s | tripod median=0.7556094744464561 (n=100)
  g_v0_w0p15: cmd 0.00 -> achieved 0.000 m/s | tripod median=0.7810810504722125 (n=100)
  g_v0_w0p3: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.8073776750401354 (n=100)
  g_v0p22_fwd_w0: cmd 0.22 -> achieved 0.235 m/s | tripod median=0.7012379816864065 (n=100)
  g_v0p22_fwd_w0p08: cmd 0.22 -> achieved 0.197 m/s | tripod median=0.6103162612769895 (n=100)
  g_v0p22_fwd_w0p15: cmd 0.22 -> achieved 0.179 m/s | tripod median=0.612254239005503 (n=100)
  g_v0p22_fwd_w0p3: cmd 0.22 -> achieved 0.169 m/s | tripod median=0.5972103798354176 (n=100)
  g_v0p22_lat_w0: cmd 0.22 -> achieved 0.234 m/s | tripod median=0.7253960365623595 (n=100)
  g_v0p22_lat_w0p08: cmd 0.22 -> achieved 0.225 m/s | tripod median=0.6746236855156447 (n=100)
  g_v0p22_lat_w0p15: cmd 0.22 -> achieved 0.213 m/s | tripod median=0.7073554036822582 (n=100)
  g_v0p22_lat_w0p3: cmd 0.22 -> achieved 0.196 m/s | tripod median=0.7384506198795692 (n=100)
  g_v0p3_fwd_w0: cmd 0.30 -> achieved 0.275 m/s | tripod median=0.4434627007250768 (n=100)
  g_v0p3_fwd_w0p08: cmd 0.30 -> achieved 0.209 m/s | tripod median=0.3681787407296667 (n=100)
  g_v0p3_fwd_w0p15: cmd 0.30 -> achieved 0.198 m/s | tripod median=0.366762724928269 (n=100)
  g_v0p3_fwd_w0p3: cmd 0.30 -> achieved 0.184 m/s | tripod median=0.35439005999312256 (n=100)
  g_v0p3_lat_w0: cmd 0.30 -> achieved 0.277 m/s | tripod median=0.6964044611591182 (n=100)
  g_v0p3_lat_w0p08: cmd 0.30 -> achieved 0.270 m/s | tripod median=0.6566725503524313 (n=100)
  g_v0p3_lat_w0p15: cmd 0.30 -> achieved 0.253 m/s | tripod median=0.6848552556968703 (n=100)
  g_v0p3_lat_w0p3: cmd 0.30 -> achieved 0.237 m/s | tripod median=0.724891406985604 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.017 initial=0.005 final=0.015 max=0.034 closed=n/a | settle=0.000 (n=2000/2000) | progress=n/a net=0.001 yaw_rate_ratio=n/a
   g_v0_w0: err rms=0.026 initial=0.015 final=0.024 max=0.058 closed=n/a | settle=0.000 (n=100/100) | progress=-0.004 net=0.001 yaw_rate_ratio=0.437
  g_v0_w0p08: err rms=0.154 initial=0.016 final=0.168 max=0.240 closed=n/a | settle=11.820 (n=35/100) | progress=0.077 net=0.012 yaw_rate_ratio=0.584
  g_v0_w0p15: err rms=0.277 initial=0.015 final=0.351 max=0.398 closed=n/a | settle=n/a (n=0/100) | progress=0.131 net=0.023 yaw_rate_ratio=0.580
  g_v0_w0p3: err rms=0.539 initial=0.013 final=0.600 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.158 net=0.050 yaw_rate_ratio=0.580
  g_v0p22_fwd_w0: err rms=0.062 initial=0.015 final=0.095 max=0.118 closed=n/a | settle=0.000 (n=99/100) | progress=-0.002 net=-0.002 yaw_rate_ratio=0.279
  g_v0p22_fwd_w0p08: err rms=0.279 initial=0.015 final=0.444 max=0.447 closed=n/a | settle=n/a (n=0/100) | progress=0.049 net=0.035 yaw_rate_ratio=0.333
  g_v0p22_fwd_w0p15: err rms=0.486 initial=0.015 final=0.600 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.062 net=0.050 yaw_rate_ratio=0.334
  g_v0p22_fwd_w0p3: err rms=0.560 initial=0.011 final=0.600 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.064 net=0.002 yaw_rate_ratio=0.350
  g_v0p22_lat_w0: err rms=0.023 initial=0.011 final=0.027 max=0.050 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
  g_v0p22_lat_w0p08: err rms=0.156 initial=0.013 final=0.167 max=0.236 closed=n/a | settle=11.660 (n=35/100) | progress=0.076 net=0.012 yaw_rate_ratio=0.525
  g_v0p22_lat_w0p15: err rms=0.271 initial=0.013 final=0.281 max=0.365 closed=n/a | settle=n/a (n=0/100) | progress=0.137 net=0.002 yaw_rate_ratio=0.544
  g_v0p22_lat_w0p3: err rms=0.546 initial=0.013 final=0.600 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.148 net=-0.047 yaw_rate_ratio=0.561
  g_v0p3_fwd_w0: err rms=0.054 initial=0.014 final=0.058 max=0.112 closed=n/a | settle=0.000 (n=99/100) | progress=-0.001 net=0.002 yaw_rate_ratio=0.506
  g_v0p3_fwd_w0p08: err rms=0.352 initial=0.016 final=0.514 max=0.534 closed=n/a | settle=n/a (n=0/100) | progress=0.041 net=0.037 yaw_rate_ratio=0.396
  g_v0p3_fwd_w0p15: err rms=0.508 initial=0.012 final=0.600 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.050 net=0.050 yaw_rate_ratio=0.386
  g_v0p3_fwd_w0p3: err rms=0.562 initial=0.014 final=0.600 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.054 net=0.050 yaw_rate_ratio=0.381
  g_v0p3_lat_w0: err rms=0.028 initial=0.019 final=0.027 max=0.059 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a
  g_v0p3_lat_w0p08: err rms=0.172 initial=0.014 final=0.196 max=0.255 closed=n/a | settle=11.760 (n=13/100) | progress=0.073 net=0.011 yaw_rate_ratio=0.461
  g_v0p3_lat_w0p15: err rms=0.360 initial=0.015 final=0.460 max=0.496 closed=n/a | settle=n/a (n=0/100) | progress=0.120 net=0.035 yaw_rate_ratio=0.501
  g_v0p3_lat_w0p3: err rms=0.550 initial=0.016 final=0.600 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.140 net=0.050 yaw_rate_ratio=0.530

body yaw by hold (window [4 s, hold_s); medians; net = net yaw rate rad/s, ratio = net / wz_cmd SIGNED (the gated turn number), wz_ratio = mean wz / wz_cmd; ripple = std of the 1-s mean wz; drift = |mean wz| on wz = 0 holds; planar = xy drift m/s on pure turns; body = along-bearing speed / |v_cmd|, cross = mean |perp| m/s):
          settle: net=n/a ratio=n/a wz_ratio=n/a ripple=n/a | drift=n/a planar=n/a | body=n/a cross=n/a (n=0)
         g_v0_w0: net=0.000 ratio=n/a wz_ratio=n/a ripple=0.021 | drift=0.002 planar=n/a | body=n/a cross=n/a (n=100)
      g_v0_w0p08: net=0.080 ratio=1.001 wz_ratio=1.014 ripple=0.030 | drift=n/a planar=0.006 | body=n/a cross=n/a (n=100)
      g_v0_w0p15: net=0.136 ratio=0.910 wz_ratio=0.916 ripple=0.029 | drift=n/a planar=0.006 | body=n/a cross=n/a (n=100)
       g_v0_w0p3: net=0.157 ratio=0.523 wz_ratio=0.524 ripple=0.026 | drift=n/a planar=0.008 | body=n/a cross=n/a (n=100)
  g_v0p22_fwd_w0: net=0.009 ratio=n/a wz_ratio=n/a ripple=0.022 | drift=0.009 planar=n/a | body=1.069 cross=0.026 (n=100)
  g_v0p22_fwd_w0p08: net=0.051 ratio=0.639 wz_ratio=0.636 ripple=0.028 | drift=n/a planar=n/a | body=0.854 cross=0.030 (n=100)
  g_v0p22_fwd_w0p15: net=0.064 ratio=0.425 wz_ratio=0.421 ripple=0.029 | drift=n/a planar=n/a | body=0.759 cross=0.029 (n=100)
  g_v0p22_fwd_w0p3: net=0.063 ratio=0.211 wz_ratio=0.208 ripple=0.025 | drift=n/a planar=n/a | body=0.740 cross=0.030 (n=100)
  g_v0p22_lat_w0: net=0.001 ratio=n/a wz_ratio=n/a ripple=0.011 | drift=0.002 planar=n/a | body=1.067 cross=0.022 (n=100)
  g_v0p22_lat_w0p08: net=0.079 ratio=0.990 wz_ratio=0.996 ripple=0.029 | drift=n/a planar=n/a | body=1.011 cross=0.039 (n=100)
  g_v0p22_lat_w0p15: net=0.151 ratio=1.009 wz_ratio=1.011 ripple=0.023 | drift=n/a planar=n/a | body=0.946 cross=0.043 (n=100)
  g_v0p22_lat_w0p3: net=0.151 ratio=0.504 wz_ratio=0.503 ripple=0.022 | drift=n/a planar=n/a | body=0.865 cross=0.055 (n=100)
   g_v0p3_fwd_w0: net=-0.005 ratio=n/a wz_ratio=n/a ripple=0.038 | drift=0.005 planar=n/a | body=0.894 cross=0.031 (n=100)
  g_v0p3_fwd_w0p08: net=0.050 ratio=0.621 wz_ratio=0.601 ripple=0.033 | drift=n/a planar=n/a | body=0.608 cross=0.038 (n=100)
  g_v0p3_fwd_w0p15: net=0.058 ratio=0.388 wz_ratio=0.375 ripple=0.031 | drift=n/a planar=n/a | body=0.577 cross=0.039 (n=100)
  g_v0p3_fwd_w0p3: net=0.061 ratio=0.202 wz_ratio=0.194 ripple=0.029 | drift=n/a planar=n/a | body=0.546 cross=0.038 (n=100)
   g_v0p3_lat_w0: net=0.002 ratio=n/a wz_ratio=n/a ripple=0.015 | drift=0.003 planar=n/a | body=0.921 cross=0.030 (n=100)
  g_v0p3_lat_w0p08: net=0.077 ratio=0.964 wz_ratio=0.965 ripple=0.025 | drift=n/a planar=n/a | body=0.895 cross=0.042 (n=100)
  g_v0p3_lat_w0p15: net=0.135 ratio=0.901 wz_ratio=0.902 ripple=0.024 | drift=n/a planar=n/a | body=0.828 cross=0.048 (n=100)
  g_v0p3_lat_w0p3: net=0.148 ratio=0.493 wz_ratio=0.493 ripple=0.024 | drift=n/a planar=n/a | body=0.756 cross=0.056 (n=100)

by category (20 waves, 2000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: n/a):
    g_v0_w0: n=100 surv=1.000 (g_v0_w0=1.00) | ratio=n/a achieved=0.001 tripod=0.800 | closed=n/a progress=-0.004 net=0.001 | drift=0.038 dir_err=n/a
  g_v0_w0p08: n=100 surv=1.000 (g_v0_w0p08=1.00) | ratio=n/a achieved=-0.004 tripod=0.756 | closed=n/a progress=0.077 net=0.012 | drift=0.042 dir_err=n/a
  g_v0_w0p15: n=100 surv=1.000 (g_v0_w0p15=1.00) | ratio=n/a achieved=0.000 tripod=0.781 | closed=n/a progress=0.131 net=0.023 | drift=0.044 dir_err=n/a
  g_v0_w0p3: n=100 surv=1.000 (g_v0_w0p3=1.00) | ratio=n/a achieved=0.001 tripod=0.807 | closed=n/a progress=0.158 net=0.050 | drift=0.050 dir_err=n/a
  g_v0p22_fwd_w0: n=100 surv=1.000 (g_v0p22_fwd_w0=1.00) | ratio=1.067 achieved=0.235 tripod=0.701 | closed=n/a progress=-0.002 net=-0.002 | drift=0.233 dir_err=0.068
  g_v0p22_fwd_w0p08: n=100 surv=1.000 (g_v0p22_fwd_w0p08=1.00) | ratio=0.897 achieved=0.197 tripod=0.610 | closed=n/a progress=0.049 net=0.035 | drift=0.200 dir_err=0.077
  g_v0p22_fwd_w0p15: n=100 surv=1.000 (g_v0p22_fwd_w0p15=1.00) | ratio=0.812 achieved=0.179 tripod=0.612 | closed=n/a progress=0.062 net=0.050 | drift=0.184 dir_err=0.071
  g_v0p22_fwd_w0p3: n=100 surv=0.990 (g_v0p22_fwd_w0p3=0.99) | ratio=0.770 achieved=0.169 tripod=0.597 | closed=n/a progress=0.064 net=0.002 | drift=0.176 dir_err=0.070
  g_v0p22_lat_w0: n=100 surv=1.000 (g_v0p22_lat_w0=1.00) | ratio=1.064 achieved=0.234 tripod=0.725 | closed=n/a progress=n/a net=0.001 | drift=0.233 dir_err=0.011
  g_v0p22_lat_w0p08: n=100 surv=1.000 (g_v0p22_lat_w0p08=1.00) | ratio=1.022 achieved=0.225 tripod=0.675 | closed=n/a progress=0.076 net=0.012 | drift=0.228 dir_err=0.032
  g_v0p22_lat_w0p15: n=100 surv=1.000 (g_v0p22_lat_w0p15=1.00) | ratio=0.967 achieved=0.213 tripod=0.707 | closed=n/a progress=0.137 net=0.002 | drift=0.219 dir_err=0.048
  g_v0p22_lat_w0p3: n=100 surv=1.000 (g_v0p22_lat_w0p3=1.00) | ratio=0.890 achieved=0.196 tripod=0.738 | closed=n/a progress=0.148 net=-0.047 | drift=0.209 dir_err=0.131
  g_v0p3_fwd_w0: n=100 surv=0.980 (g_v0p3_fwd_w0=0.98) | ratio=0.917 achieved=0.275 tripod=0.443 | closed=n/a progress=-0.001 net=0.002 | drift=0.275 dir_err=0.040
  g_v0p3_fwd_w0p08: n=100 surv=0.880 (g_v0p3_fwd_w0p08=0.88) | ratio=0.697 achieved=0.209 tripod=0.368 | closed=n/a progress=0.041 net=0.037 | drift=0.226 dir_err=0.064
  g_v0p3_fwd_w0p15: n=100 surv=0.890 (g_v0p3_fwd_w0p15=0.89) | ratio=0.660 achieved=0.198 tripod=0.367 | closed=n/a progress=0.050 net=0.050 | drift=0.215 dir_err=0.066
  g_v0p3_fwd_w0p3: n=100 surv=0.890 (g_v0p3_fwd_w0p3=0.89) | ratio=0.613 achieved=0.184 tripod=0.354 | closed=n/a progress=0.054 net=0.050 | drift=0.207 dir_err=0.056
  g_v0p3_lat_w0: n=100 surv=1.000 (g_v0p3_lat_w0=1.00) | ratio=0.923 achieved=0.277 tripod=0.696 | closed=n/a progress=n/a net=0.000 | drift=0.277 dir_err=0.012
  g_v0p3_lat_w0p08: n=100 surv=1.000 (g_v0p3_lat_w0p08=1.00) | ratio=0.900 achieved=0.270 tripod=0.657 | closed=n/a progress=0.073 net=0.011 | drift=0.273 dir_err=0.045
  g_v0p3_lat_w0p15: n=100 surv=1.000 (g_v0p3_lat_w0p15=1.00) | ratio=0.845 achieved=0.253 tripod=0.685 | closed=n/a progress=0.120 net=0.035 | drift=0.260 dir_err=0.084
  g_v0p3_lat_w0p3: n=100 surv=1.000 (g_v0p3_lat_w0p3=1.00) | ratio=0.789 achieved=0.237 tripod=0.725 | closed=n/a progress=0.140 net=0.050 | drift=0.247 dir_err=0.127

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
    g_v0_w0: net_yaw_rate=0.000 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.021 heading_drift=0.002 planar_drift=n/a body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  g_v0_w0p08: net_yaw_rate=0.080 net_yaw_ratio=1.001 wz_ratio=1.014 yaw_ripple=0.030 heading_drift=n/a planar_drift=0.006 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  g_v0_w0p15: net_yaw_rate=0.136 net_yaw_ratio=0.910 wz_ratio=0.916 yaw_ripple=0.029 heading_drift=n/a planar_drift=0.006 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  g_v0_w0p3: net_yaw_rate=0.157 net_yaw_ratio=0.523 wz_ratio=0.524 yaw_ripple=0.026 heading_drift=n/a planar_drift=0.008 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  g_v0p22_fwd_w0: net_yaw_rate=0.009 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.022 heading_drift=0.009 planar_drift=n/a body_ratio=1.069 cross_body=0.026 adapter_clipped_frac=n/a
  g_v0p22_fwd_w0p08: net_yaw_rate=0.051 net_yaw_ratio=0.639 wz_ratio=0.636 yaw_ripple=0.028 heading_drift=n/a planar_drift=n/a body_ratio=0.854 cross_body=0.030 adapter_clipped_frac=n/a
  g_v0p22_fwd_w0p15: net_yaw_rate=0.064 net_yaw_ratio=0.425 wz_ratio=0.421 yaw_ripple=0.029 heading_drift=n/a planar_drift=n/a body_ratio=0.759 cross_body=0.029 adapter_clipped_frac=n/a
  g_v0p22_fwd_w0p3: net_yaw_rate=0.063 net_yaw_ratio=0.211 wz_ratio=0.208 yaw_ripple=0.025 heading_drift=n/a planar_drift=n/a body_ratio=0.740 cross_body=0.030 adapter_clipped_frac=n/a
  g_v0p22_lat_w0: net_yaw_rate=0.001 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.011 heading_drift=0.002 planar_drift=n/a body_ratio=1.067 cross_body=0.022 adapter_clipped_frac=n/a
  g_v0p22_lat_w0p08: net_yaw_rate=0.079 net_yaw_ratio=0.990 wz_ratio=0.996 yaw_ripple=0.029 heading_drift=n/a planar_drift=n/a body_ratio=1.011 cross_body=0.039 adapter_clipped_frac=n/a
  g_v0p22_lat_w0p15: net_yaw_rate=0.151 net_yaw_ratio=1.009 wz_ratio=1.011 yaw_ripple=0.023 heading_drift=n/a planar_drift=n/a body_ratio=0.946 cross_body=0.043 adapter_clipped_frac=n/a
  g_v0p22_lat_w0p3: net_yaw_rate=0.151 net_yaw_ratio=0.504 wz_ratio=0.503 yaw_ripple=0.022 heading_drift=n/a planar_drift=n/a body_ratio=0.865 cross_body=0.055 adapter_clipped_frac=n/a
  g_v0p3_fwd_w0: net_yaw_rate=-0.005 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.038 heading_drift=0.005 planar_drift=n/a body_ratio=0.894 cross_body=0.031 adapter_clipped_frac=n/a
  g_v0p3_fwd_w0p08: net_yaw_rate=0.050 net_yaw_ratio=0.621 wz_ratio=0.601 yaw_ripple=0.033 heading_drift=n/a planar_drift=n/a body_ratio=0.608 cross_body=0.038 adapter_clipped_frac=n/a
  g_v0p3_fwd_w0p15: net_yaw_rate=0.058 net_yaw_ratio=0.388 wz_ratio=0.375 yaw_ripple=0.031 heading_drift=n/a planar_drift=n/a body_ratio=0.577 cross_body=0.039 adapter_clipped_frac=n/a
  g_v0p3_fwd_w0p3: net_yaw_rate=0.061 net_yaw_ratio=0.202 wz_ratio=0.194 yaw_ripple=0.029 heading_drift=n/a planar_drift=n/a body_ratio=0.546 cross_body=0.038 adapter_clipped_frac=n/a
  g_v0p3_lat_w0: net_yaw_rate=0.002 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.015 heading_drift=0.003 planar_drift=n/a body_ratio=0.921 cross_body=0.030 adapter_clipped_frac=n/a
  g_v0p3_lat_w0p08: net_yaw_rate=0.077 net_yaw_ratio=0.964 wz_ratio=0.965 yaw_ripple=0.025 heading_drift=n/a planar_drift=n/a body_ratio=0.895 cross_body=0.042 adapter_clipped_frac=n/a
  g_v0p3_lat_w0p15: net_yaw_rate=0.135 net_yaw_ratio=0.901 wz_ratio=0.902 yaw_ripple=0.024 heading_drift=n/a planar_drift=n/a body_ratio=0.828 cross_body=0.048 adapter_clipped_frac=n/a
  g_v0p3_lat_w0p3: net_yaw_rate=0.148 net_yaw_ratio=0.493 wz_ratio=0.493 yaw_ripple=0.024 heading_drift=n/a planar_drift=n/a body_ratio=0.756 cross_body=0.056 adapter_clipped_frac=n/a

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
