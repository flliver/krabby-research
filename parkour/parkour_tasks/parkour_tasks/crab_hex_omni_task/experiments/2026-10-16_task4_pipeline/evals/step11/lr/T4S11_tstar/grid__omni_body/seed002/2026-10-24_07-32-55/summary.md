```
=== crab-hex gait eval ===
scenario   : grid__omni_body  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt
episodes   : 2000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7116931240345563  p25=0.6562252005559279  p75=0.7562450995647831
tippy_tap_fraction median=0.4569664902998236
slip_ratio         median=0.24865754696441955
tracking_ratio     median=0.8938276425434828  (achieved/commanded vx, walking holds; n=1600)
schedule_completion_rate=1.0
terminations={'schedule_complete': 2000}
facing_error_rms   median=0.21110671168532424  (rad, whole episode; n=2000)

by hold:
    settle: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7680647207804914 (n=1947)
   g_v0_w0: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.787077889479137 (n=100)
  g_v0_w0p08: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.7807972202472844 (n=100)
  g_v0_w0p15: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.7819102379578842 (n=100)
  g_v0_w0p3: cmd 0.00 -> achieved -0.011 m/s | tripod median=0.7845431646124168 (n=100)
  g_v0p22_fwd_w0: cmd 0.22 -> achieved 0.227 m/s | tripod median=0.701241496545975 (n=100)
  g_v0p22_fwd_w0p08: cmd 0.22 -> achieved 0.197 m/s | tripod median=0.6909142609728851 (n=100)
  g_v0p22_fwd_w0p15: cmd 0.22 -> achieved 0.173 m/s | tripod median=0.6621306111836831 (n=100)
  g_v0p22_fwd_w0p3: cmd 0.22 -> achieved 0.161 m/s | tripod median=0.6597703959456642 (n=100)
  g_v0p22_lat_w0: cmd 0.22 -> achieved 0.235 m/s | tripod median=0.6779721433458481 (n=100)
  g_v0p22_lat_w0p08: cmd 0.22 -> achieved 0.226 m/s | tripod median=0.6933271571974926 (n=100)
  g_v0p22_lat_w0p15: cmd 0.22 -> achieved 0.219 m/s | tripod median=0.7234514613587053 (n=100)
  g_v0p22_lat_w0p3: cmd 0.22 -> achieved 0.198 m/s | tripod median=0.7608269586018575 (n=100)
  g_v0p3_fwd_w0: cmd 0.30 -> achieved 0.278 m/s | tripod median=0.5737667344341434 (n=100)
  g_v0p3_fwd_w0p08: cmd 0.30 -> achieved 0.241 m/s | tripod median=0.46627324113620483 (n=100)
  g_v0p3_fwd_w0p15: cmd 0.30 -> achieved 0.224 m/s | tripod median=0.4262149719203848 (n=100)
  g_v0p3_fwd_w0p3: cmd 0.30 -> achieved 0.209 m/s | tripod median=0.41134751506525413 (n=100)
  g_v0p3_lat_w0: cmd 0.30 -> achieved 0.283 m/s | tripod median=0.7180591221572847 (n=100)
  g_v0p3_lat_w0p08: cmd 0.30 -> achieved 0.270 m/s | tripod median=0.7244380418362131 (n=100)
  g_v0p3_lat_w0p15: cmd 0.30 -> achieved 0.260 m/s | tripod median=0.7594209559793053 (n=100)
  g_v0p3_lat_w0p3: cmd 0.30 -> achieved 0.239 m/s | tripod median=0.752165026958047 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.014 initial=0.008 final=0.012 max=0.028 closed=n/a | settle=0.000 (n=2000/2000) | progress=n/a net=0.001 yaw_rate_ratio=n/a
   g_v0_w0: err rms=0.024 initial=0.013 final=0.028 max=0.055 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
  g_v0_w0p08: err rms=0.129 initial=0.013 final=0.142 max=0.208 closed=n/a | settle=11.680 (n=61/100) | progress=0.078 net=0.010 yaw_rate_ratio=0.671
  g_v0_w0p15: err rms=0.185 initial=0.011 final=0.184 max=0.277 closed=n/a | settle=11.780 (n=23/100) | progress=0.146 net=-0.011 yaw_rate_ratio=0.646
  g_v0_w0p3: err rms=0.529 initial=0.013 final=0.600 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.182 net=-0.047 yaw_rate_ratio=0.671
  g_v0p22_fwd_w0: err rms=0.045 initial=0.014 final=0.030 max=0.106 closed=n/a | settle=0.000 (n=100/100) | progress=-0.001 net=0.002 yaw_rate_ratio=0.332
  g_v0p22_fwd_w0p08: err rms=0.235 initial=0.013 final=0.369 max=0.380 closed=n/a | settle=n/a (n=0/100) | progress=0.054 net=0.029 yaw_rate_ratio=0.287
  g_v0p22_fwd_w0p15: err rms=0.479 initial=0.010 final=0.600 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.064 net=0.050 yaw_rate_ratio=0.315
  g_v0p22_fwd_w0p3: err rms=0.556 initial=0.013 final=0.600 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.065 net=0.050 yaw_rate_ratio=0.336
  g_v0p22_lat_w0: err rms=0.020 initial=0.013 final=0.016 max=0.045 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
  g_v0p22_lat_w0p08: err rms=0.153 initial=0.011 final=0.153 max=0.231 closed=n/a | settle=11.700 (n=48/100) | progress=0.077 net=-0.009 yaw_rate_ratio=0.546
  g_v0p22_lat_w0p15: err rms=0.245 initial=0.014 final=0.269 max=0.346 closed=n/a | settle=n/a (n=0/100) | progress=0.138 net=-0.014 yaw_rate_ratio=0.565
  g_v0p22_lat_w0p3: err rms=0.539 initial=0.010 final=0.600 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.167 net=-0.047 yaw_rate_ratio=0.617
  g_v0p3_fwd_w0: err rms=0.038 initial=0.013 final=0.042 max=0.074 closed=n/a | settle=0.000 (n=100/100) | progress=0.000 net=0.000 yaw_rate_ratio=0.374
  g_v0p3_fwd_w0p08: err rms=0.382 initial=0.011 final=0.582 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.033 net=0.000 yaw_rate_ratio=0.353
  g_v0p3_fwd_w0p15: err rms=0.509 initial=0.012 final=0.600 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.040 net=-0.048 yaw_rate_ratio=0.341
  g_v0p3_fwd_w0p3: err rms=0.560 initial=0.013 final=0.600 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.042 net=-0.048 yaw_rate_ratio=0.342
  g_v0p3_lat_w0: err rms=0.017 initial=0.012 final=0.014 max=0.043 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a
  g_v0p3_lat_w0p08: err rms=0.149 initial=0.013 final=0.146 max=0.223 closed=n/a | settle=11.670 (n=60/100) | progress=0.078 net=-0.008 yaw_rate_ratio=0.551
  g_v0p3_lat_w0p15: err rms=0.290 initial=0.010 final=0.362 max=0.413 closed=n/a | settle=n/a (n=0/100) | progress=0.129 net=-0.025 yaw_rate_ratio=0.533
  g_v0p3_lat_w0p3: err rms=0.544 initial=0.011 final=0.600 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.152 net=-0.047 yaw_rate_ratio=0.565

body yaw by hold (window [4 s, hold_s); medians; net = net yaw rate rad/s, ratio = net / wz_cmd SIGNED (the gated turn number), wz_ratio = mean wz / wz_cmd; ripple = std of the 1-s mean wz; drift = |mean wz| on wz = 0 holds; planar = xy drift m/s on pure turns; body = along-bearing speed / |v_cmd|, cross = mean |perp| m/s):
          settle: net=n/a ratio=n/a wz_ratio=n/a ripple=n/a | drift=n/a planar=n/a | body=n/a cross=n/a (n=0)
         g_v0_w0: net=0.001 ratio=n/a wz_ratio=n/a ripple=0.015 | drift=0.003 planar=n/a | body=n/a cross=n/a (n=100)
      g_v0_w0p08: net=0.080 ratio=1.002 wz_ratio=1.010 ripple=0.032 | drift=n/a planar=0.004 | body=n/a cross=n/a (n=100)
      g_v0_w0p15: net=0.148 ratio=0.986 wz_ratio=0.995 ripple=0.029 | drift=n/a planar=0.006 | body=n/a cross=n/a (n=100)
       g_v0_w0p3: net=0.187 ratio=0.622 wz_ratio=0.624 ripple=0.028 | drift=n/a planar=0.015 | body=n/a cross=n/a (n=100)
  g_v0p22_fwd_w0: net=-0.005 ratio=n/a wz_ratio=n/a ripple=0.041 | drift=0.006 planar=n/a | body=1.031 cross=0.025 (n=100)
  g_v0p22_fwd_w0p08: net=0.053 ratio=0.665 wz_ratio=0.658 ripple=0.027 | drift=n/a planar=n/a | body=0.855 cross=0.026 (n=100)
  g_v0p22_fwd_w0p15: net=0.061 ratio=0.407 wz_ratio=0.404 ripple=0.040 | drift=n/a planar=n/a | body=0.727 cross=0.027 (n=100)
  g_v0p22_fwd_w0p3: net=0.058 ratio=0.193 wz_ratio=0.191 ripple=0.032 | drift=n/a planar=n/a | body=0.693 cross=0.026 (n=100)
  g_v0p22_lat_w0: net=-0.001 ratio=n/a wz_ratio=n/a ripple=0.016 | drift=0.002 planar=n/a | body=1.071 cross=0.024 (n=100)
  g_v0p22_lat_w0p08: net=0.085 ratio=1.061 wz_ratio=1.071 ripple=0.031 | drift=n/a planar=n/a | body=1.019 cross=0.038 (n=100)
  g_v0p22_lat_w0p15: net=0.151 ratio=1.006 wz_ratio=1.009 ripple=0.028 | drift=n/a planar=n/a | body=0.991 cross=0.037 (n=100)
  g_v0p22_lat_w0p3: net=0.176 ratio=0.585 wz_ratio=0.587 ripple=0.029 | drift=n/a planar=n/a | body=0.890 cross=0.052 (n=100)
   g_v0p3_fwd_w0: net=-0.002 ratio=n/a wz_ratio=n/a ripple=0.020 | drift=0.003 planar=n/a | body=0.922 cross=0.025 (n=100)
  g_v0p3_fwd_w0p08: net=0.036 ratio=0.456 wz_ratio=0.449 ripple=0.035 | drift=n/a planar=n/a | body=0.763 cross=0.030 (n=100)
  g_v0p3_fwd_w0p15: net=0.042 ratio=0.283 wz_ratio=0.277 ripple=0.033 | drift=n/a planar=n/a | body=0.705 cross=0.031 (n=100)
  g_v0p3_fwd_w0p3: net=0.041 ratio=0.136 wz_ratio=0.132 ripple=0.031 | drift=n/a planar=n/a | body=0.674 cross=0.033 (n=100)
   g_v0p3_lat_w0: net=0.001 ratio=n/a wz_ratio=n/a ripple=0.013 | drift=0.002 planar=n/a | body=0.942 cross=0.028 (n=100)
  g_v0p3_lat_w0p08: net=0.084 ratio=1.053 wz_ratio=1.054 ripple=0.026 | drift=n/a planar=n/a | body=0.891 cross=0.036 (n=100)
  g_v0p3_lat_w0p15: net=0.142 ratio=0.948 wz_ratio=0.952 ripple=0.026 | drift=n/a planar=n/a | body=0.850 cross=0.038 (n=100)
  g_v0p3_lat_w0p3: net=0.160 ratio=0.533 wz_ratio=0.533 ripple=0.033 | drift=n/a planar=n/a | body=0.781 cross=0.054 (n=100)

by category (20 waves, 2000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: n/a):
    g_v0_w0: n=100 surv=1.000 (g_v0_w0=1.00) | ratio=n/a achieved=0.004 tripod=0.787 | closed=n/a progress=n/a net=0.001 | drift=0.034 dir_err=n/a
  g_v0_w0p08: n=100 surv=1.000 (g_v0_w0p08=1.00) | ratio=n/a achieved=0.002 tripod=0.781 | closed=n/a progress=0.078 net=0.010 | drift=0.039 dir_err=n/a
  g_v0_w0p15: n=100 surv=1.000 (g_v0_w0p15=1.00) | ratio=n/a achieved=0.002 tripod=0.782 | closed=n/a progress=0.146 net=-0.011 | drift=0.041 dir_err=n/a
  g_v0_w0p3: n=100 surv=1.000 (g_v0_w0p3=1.00) | ratio=n/a achieved=-0.011 tripod=0.785 | closed=n/a progress=0.182 net=-0.047 | drift=0.054 dir_err=n/a
  g_v0p22_fwd_w0: n=100 surv=1.000 (g_v0p22_fwd_w0=1.00) | ratio=1.034 achieved=0.227 tripod=0.701 | closed=n/a progress=-0.001 net=0.002 | drift=0.227 dir_err=0.009
  g_v0p22_fwd_w0p08: n=100 surv=1.000 (g_v0p22_fwd_w0p08=1.00) | ratio=0.897 achieved=0.197 tripod=0.691 | closed=n/a progress=0.054 net=0.029 | drift=0.201 dir_err=0.021
  g_v0p22_fwd_w0p15: n=100 surv=1.000 (g_v0p22_fwd_w0p15=1.00) | ratio=0.785 achieved=0.173 tripod=0.662 | closed=n/a progress=0.064 net=0.050 | drift=0.179 dir_err=0.017
  g_v0p22_fwd_w0p3: n=100 surv=1.000 (g_v0p22_fwd_w0p3=1.00) | ratio=0.731 achieved=0.161 tripod=0.660 | closed=n/a progress=0.065 net=0.050 | drift=0.169 dir_err=0.014
  g_v0p22_lat_w0: n=100 surv=1.000 (g_v0p22_lat_w0=1.00) | ratio=1.066 achieved=0.235 tripod=0.678 | closed=n/a progress=n/a net=0.001 | drift=0.236 dir_err=0.008
  g_v0p22_lat_w0p08: n=100 surv=1.000 (g_v0p22_lat_w0p08=1.00) | ratio=1.027 achieved=0.226 tripod=0.693 | closed=n/a progress=0.077 net=-0.009 | drift=0.229 dir_err=0.041
  g_v0p22_lat_w0p15: n=100 surv=1.000 (g_v0p22_lat_w0p15=1.00) | ratio=0.997 achieved=0.219 tripod=0.723 | closed=n/a progress=0.138 net=-0.014 | drift=0.224 dir_err=0.036
  g_v0p22_lat_w0p3: n=100 surv=1.000 (g_v0p22_lat_w0p3=1.00) | ratio=0.900 achieved=0.198 tripod=0.761 | closed=n/a progress=0.167 net=-0.047 | drift=0.211 dir_err=0.025
  g_v0p3_fwd_w0: n=100 surv=1.000 (g_v0p3_fwd_w0=1.00) | ratio=0.925 achieved=0.278 tripod=0.574 | closed=n/a progress=0.000 net=0.000 | drift=0.279 dir_err=0.011
  g_v0p3_fwd_w0p08: n=100 surv=1.000 (g_v0p3_fwd_w0p08=1.00) | ratio=0.803 achieved=0.241 tripod=0.466 | closed=n/a progress=0.033 net=0.000 | drift=0.246 dir_err=0.012
  g_v0p3_fwd_w0p15: n=100 surv=1.000 (g_v0p3_fwd_w0p15=1.00) | ratio=0.745 achieved=0.224 tripod=0.426 | closed=n/a progress=0.040 net=-0.048 | drift=0.230 dir_err=0.015
  g_v0p3_fwd_w0p3: n=100 surv=1.000 (g_v0p3_fwd_w0p3=1.00) | ratio=0.698 achieved=0.209 tripod=0.411 | closed=n/a progress=0.042 net=-0.048 | drift=0.220 dir_err=0.014
  g_v0p3_lat_w0: n=100 surv=1.000 (g_v0p3_lat_w0=1.00) | ratio=0.942 achieved=0.283 tripod=0.718 | closed=n/a progress=n/a net=0.000 | drift=0.284 dir_err=0.012
  g_v0p3_lat_w0p08: n=100 surv=1.000 (g_v0p3_lat_w0p08=1.00) | ratio=0.901 achieved=0.270 tripod=0.724 | closed=n/a progress=0.078 net=-0.008 | drift=0.273 dir_err=0.043
  g_v0p3_lat_w0p15: n=100 surv=1.000 (g_v0p3_lat_w0p15=1.00) | ratio=0.868 achieved=0.260 tripod=0.759 | closed=n/a progress=0.129 net=-0.025 | drift=0.265 dir_err=0.036
  g_v0p3_lat_w0p3: n=100 surv=1.000 (g_v0p3_lat_w0p3=1.00) | ratio=0.796 achieved=0.239 tripod=0.752 | closed=n/a progress=0.152 net=-0.047 | drift=0.251 dir_err=0.026

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
    g_v0_w0: net_yaw_rate=0.001 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.015 heading_drift=0.003 planar_drift=n/a body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  g_v0_w0p08: net_yaw_rate=0.080 net_yaw_ratio=1.002 wz_ratio=1.010 yaw_ripple=0.032 heading_drift=n/a planar_drift=0.004 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  g_v0_w0p15: net_yaw_rate=0.148 net_yaw_ratio=0.986 wz_ratio=0.995 yaw_ripple=0.029 heading_drift=n/a planar_drift=0.006 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  g_v0_w0p3: net_yaw_rate=0.187 net_yaw_ratio=0.622 wz_ratio=0.624 yaw_ripple=0.028 heading_drift=n/a planar_drift=0.015 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  g_v0p22_fwd_w0: net_yaw_rate=-0.005 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.041 heading_drift=0.006 planar_drift=n/a body_ratio=1.031 cross_body=0.025 adapter_clipped_frac=n/a
  g_v0p22_fwd_w0p08: net_yaw_rate=0.053 net_yaw_ratio=0.665 wz_ratio=0.658 yaw_ripple=0.027 heading_drift=n/a planar_drift=n/a body_ratio=0.855 cross_body=0.026 adapter_clipped_frac=n/a
  g_v0p22_fwd_w0p15: net_yaw_rate=0.061 net_yaw_ratio=0.407 wz_ratio=0.404 yaw_ripple=0.040 heading_drift=n/a planar_drift=n/a body_ratio=0.727 cross_body=0.027 adapter_clipped_frac=n/a
  g_v0p22_fwd_w0p3: net_yaw_rate=0.058 net_yaw_ratio=0.193 wz_ratio=0.191 yaw_ripple=0.032 heading_drift=n/a planar_drift=n/a body_ratio=0.693 cross_body=0.026 adapter_clipped_frac=n/a
  g_v0p22_lat_w0: net_yaw_rate=-0.001 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.016 heading_drift=0.002 planar_drift=n/a body_ratio=1.071 cross_body=0.024 adapter_clipped_frac=n/a
  g_v0p22_lat_w0p08: net_yaw_rate=0.085 net_yaw_ratio=1.061 wz_ratio=1.071 yaw_ripple=0.031 heading_drift=n/a planar_drift=n/a body_ratio=1.019 cross_body=0.038 adapter_clipped_frac=n/a
  g_v0p22_lat_w0p15: net_yaw_rate=0.151 net_yaw_ratio=1.006 wz_ratio=1.009 yaw_ripple=0.028 heading_drift=n/a planar_drift=n/a body_ratio=0.991 cross_body=0.037 adapter_clipped_frac=n/a
  g_v0p22_lat_w0p3: net_yaw_rate=0.176 net_yaw_ratio=0.585 wz_ratio=0.587 yaw_ripple=0.029 heading_drift=n/a planar_drift=n/a body_ratio=0.890 cross_body=0.052 adapter_clipped_frac=n/a
  g_v0p3_fwd_w0: net_yaw_rate=-0.002 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.020 heading_drift=0.003 planar_drift=n/a body_ratio=0.922 cross_body=0.025 adapter_clipped_frac=n/a
  g_v0p3_fwd_w0p08: net_yaw_rate=0.036 net_yaw_ratio=0.456 wz_ratio=0.449 yaw_ripple=0.035 heading_drift=n/a planar_drift=n/a body_ratio=0.763 cross_body=0.030 adapter_clipped_frac=n/a
  g_v0p3_fwd_w0p15: net_yaw_rate=0.042 net_yaw_ratio=0.283 wz_ratio=0.277 yaw_ripple=0.033 heading_drift=n/a planar_drift=n/a body_ratio=0.705 cross_body=0.031 adapter_clipped_frac=n/a
  g_v0p3_fwd_w0p3: net_yaw_rate=0.041 net_yaw_ratio=0.136 wz_ratio=0.132 yaw_ripple=0.031 heading_drift=n/a planar_drift=n/a body_ratio=0.674 cross_body=0.033 adapter_clipped_frac=n/a
  g_v0p3_lat_w0: net_yaw_rate=0.001 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.013 heading_drift=0.002 planar_drift=n/a body_ratio=0.942 cross_body=0.028 adapter_clipped_frac=n/a
  g_v0p3_lat_w0p08: net_yaw_rate=0.084 net_yaw_ratio=1.053 wz_ratio=1.054 yaw_ripple=0.026 heading_drift=n/a planar_drift=n/a body_ratio=0.891 cross_body=0.036 adapter_clipped_frac=n/a
  g_v0p3_lat_w0p15: net_yaw_rate=0.142 net_yaw_ratio=0.948 wz_ratio=0.952 yaw_ripple=0.026 heading_drift=n/a planar_drift=n/a body_ratio=0.850 cross_body=0.038 adapter_clipped_frac=n/a
  g_v0p3_lat_w0p3: net_yaw_rate=0.160 net_yaw_ratio=0.533 wz_ratio=0.533 yaw_ripple=0.033 heading_drift=n/a planar_drift=n/a body_ratio=0.781 cross_body=0.054 adapter_clipped_frac=n/a

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
