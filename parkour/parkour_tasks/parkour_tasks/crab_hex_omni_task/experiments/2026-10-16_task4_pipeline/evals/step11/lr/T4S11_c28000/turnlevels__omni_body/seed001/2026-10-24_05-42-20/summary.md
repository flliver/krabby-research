```
=== crab-hex gait eval ===
scenario   : turnlevels__omni_body  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_28000.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7595327438332269  p25=0.7359415834803171  p75=0.7820366526451114
tippy_tap_fraction median=0.42067124332570555
slip_ratio         median=0.24692271680380343
tracking_ratio     median=None  (achieved/commanded vx, walking holds; n=0)
schedule_completion_rate=1.0
terminations={'schedule_complete': 1000}
facing_error_rms   median=0.19696568555710842  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.6594689569810466 (n=957)
  turn_l_w_min: cmd 0.00 -> achieved -0.001 m/s | tripod median=0.7610894829618049 (n=100)
  turn_r_w_min: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.747527092768921 (n=100)
  turn_l_0p08: cmd 0.00 -> achieved -0.002 m/s | tripod median=0.7592779981076576 (n=100)
  turn_r_0p08: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.7526168421449124 (n=100)
  turn_l_0p15: cmd 0.00 -> achieved -0.002 m/s | tripod median=0.7614893202376682 (n=100)
  turn_r_0p15: cmd 0.00 -> achieved 0.008 m/s | tripod median=0.7832203778248203 (n=100)
  turn_l_ceiling: cmd 0.00 -> achieved -0.002 m/s | tripod median=0.7795730475045407 (n=100)
  turn_r_ceiling: cmd 0.00 -> achieved 0.006 m/s | tripod median=0.7933847601929743 (n=100)
  turn_l_w_max: cmd 0.00 -> achieved -0.007 m/s | tripod median=0.7829740467499395 (n=100)
  turn_r_w_max: cmd 0.00 -> achieved 0.008 m/s | tripod median=0.8050839626224446 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.015 initial=0.008 final=0.018 max=0.031 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.000 yaw_rate_ratio=n/a
  turn_l_w_min: err rms=0.114 initial=0.016 final=0.126 max=0.196 closed=n/a | settle=11.360 (n=78/100) | progress=0.048 net=0.010 yaw_rate_ratio=0.584
  turn_r_w_min: err rms=0.127 initial=0.018 final=0.130 max=0.213 closed=n/a | settle=11.250 (n=68/100) | progress=0.048 net=-0.008 yaw_rate_ratio=0.528
  turn_l_0p08: err rms=0.136 initial=0.017 final=0.159 max=0.222 closed=n/a | settle=11.580 (n=42/100) | progress=0.075 net=0.012 yaw_rate_ratio=0.626
  turn_r_0p08: err rms=0.147 initial=0.019 final=0.153 max=0.238 closed=n/a | settle=11.580 (n=45/100) | progress=0.077 net=-0.008 yaw_rate_ratio=0.553
  turn_l_0p15: err rms=0.221 initial=0.014 final=0.261 max=0.326 closed=n/a | settle=11.780 (n=4/100) | progress=0.139 net=0.020 yaw_rate_ratio=0.604
  turn_r_0p15: err rms=0.259 initial=0.017 final=0.313 max=0.387 closed=n/a | settle=n/a (n=0/100) | progress=0.134 net=-0.022 yaw_rate_ratio=0.594
  turn_l_ceiling: err rms=0.243 initial=0.018 final=0.282 max=0.339 closed=n/a | settle=n/a (n=0/100) | progress=0.155 net=0.020 yaw_rate_ratio=0.617
  turn_r_ceiling: err rms=0.295 initial=0.017 final=0.372 max=0.428 closed=n/a | settle=n/a (n=0/100) | progress=0.147 net=-0.026 yaw_rate_ratio=0.615
  turn_l_w_max: err rms=0.388 initial=0.017 final=0.554 max=0.569 closed=n/a | settle=n/a (n=0/100) | progress=0.173 net=0.043 yaw_rate_ratio=0.650
  turn_r_w_max: err rms=0.410 initial=0.018 final=0.552 max=0.590 closed=n/a | settle=n/a (n=0/100) | progress=0.173 net=-0.043 yaw_rate_ratio=0.643

body yaw by hold (window [4 s, hold_s); medians; net = net yaw rate rad/s, ratio = net / wz_cmd SIGNED (the gated turn number), wz_ratio = mean wz / wz_cmd; ripple = std of the 1-s mean wz; drift = |mean wz| on wz = 0 holds; planar = xy drift m/s on pure turns; body = along-bearing speed / |v_cmd|, cross = mean |perp| m/s):
          settle: net=n/a ratio=n/a wz_ratio=n/a ripple=n/a | drift=n/a planar=n/a | body=n/a cross=n/a (n=0)
    turn_l_w_min: net=0.049 ratio=0.971 wz_ratio=0.976 ripple=0.032 | drift=n/a planar=0.009 | body=n/a cross=n/a (n=100)
    turn_r_w_min: net=-0.050 ratio=1.005 wz_ratio=1.009 ripple=0.036 | drift=n/a planar=0.006 | body=n/a cross=n/a (n=100)
     turn_l_0p08: net=0.077 ratio=0.968 wz_ratio=0.974 ripple=0.035 | drift=n/a planar=0.010 | body=n/a cross=n/a (n=100)
     turn_r_0p08: net=-0.083 ratio=1.037 wz_ratio=1.039 ripple=0.039 | drift=n/a planar=0.005 | body=n/a cross=n/a (n=100)
     turn_l_0p15: net=0.143 ratio=0.957 wz_ratio=0.962 ripple=0.035 | drift=n/a planar=0.009 | body=n/a cross=n/a (n=100)
     turn_r_0p15: net=-0.139 ratio=0.929 wz_ratio=0.930 ripple=0.053 | drift=n/a planar=0.011 | body=n/a cross=n/a (n=100)
  turn_l_ceiling: net=0.161 ratio=0.959 wz_ratio=0.963 ripple=0.026 | drift=n/a planar=0.009 | body=n/a cross=n/a (n=100)
  turn_r_ceiling: net=-0.155 ratio=0.923 wz_ratio=0.927 ripple=0.041 | drift=n/a planar=0.010 | body=n/a cross=n/a (n=100)
    turn_l_w_max: net=0.183 ratio=0.871 wz_ratio=0.874 ripple=0.022 | drift=n/a planar=0.014 | body=n/a cross=n/a (n=100)
    turn_r_w_max: net=-0.183 ratio=0.874 wz_ratio=0.876 ripple=0.029 | drift=n/a planar=0.011 | body=n/a cross=n/a (n=100)

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: n/a):
  turn_w_min: n=200 surv=1.000 (turn_l_w_min=1.00 turn_r_w_min=1.00) | ratio=n/a achieved=0.001 tripod=0.754 | closed=n/a progress=0.048 net=-0.004 | drift=0.040 dir_err=n/a
  turn_0p08: n=200 surv=1.000 (turn_l_0p08=1.00 turn_r_0p08=1.00) | ratio=n/a achieved=0.001 tripod=0.756 | closed=n/a progress=0.076 net=-0.005 | drift=0.042 dir_err=n/a
  turn_0p15: n=200 surv=1.000 (turn_l_0p15=1.00 turn_r_0p15=1.00) | ratio=n/a achieved=0.003 tripod=0.774 | closed=n/a progress=0.136 net=-0.010 | drift=0.043 dir_err=n/a
  turn_ceiling: n=200 surv=1.000 (turn_l_ceiling=1.00 turn_r_ceiling=1.00) | ratio=n/a achieved=0.003 tripod=0.785 | closed=n/a progress=0.151 net=-0.019 | drift=0.044 dir_err=n/a
  turn_w_max: n=200 surv=1.000 (turn_l_w_max=1.00 turn_r_w_max=1.00) | ratio=n/a achieved=-0.000 tripod=0.792 | closed=n/a progress=0.173 net=-0.038 | drift=0.048 dir_err=n/a

motion quality by category (own hold, whole hold; medians; loaded = foot force > 50 N; per-m / per-rad over net displacement / |net yaw|):
  turn_w_min: 
  turn_0p08: 
  turn_0p15: 
  turn_ceiling: 
  turn_w_max: 

rigidity by category (own hold, whole hold; medians; step = unloaded run >= 5 env steps, tap = shorter; loaded = foot force > 50 N; hz_speed_slope / offset_R are one statistic over the category's episodes):
  turn_w_min: 
  turn_0p08: 
  turn_0p15: 
  turn_ceiling: 
  turn_w_max: 

body yaw by category (own hold, window [4 s, hold_s); medians pooled over both hands; per-hand values are in aggregate.suite.by_category[c].by_hand):
  turn_w_min: net_yaw_rate=-0.001 net_yaw_ratio=0.988 wz_ratio=0.988 yaw_ripple=0.033 heading_drift=n/a planar_drift=0.007 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  turn_0p08: net_yaw_rate=-0.002 net_yaw_ratio=1.002 wz_ratio=1.005 yaw_ripple=0.037 heading_drift=n/a planar_drift=0.008 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  turn_0p15: net_yaw_rate=0.012 net_yaw_ratio=0.944 wz_ratio=0.948 yaw_ripple=0.041 heading_drift=n/a planar_drift=0.010 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  turn_ceiling: net_yaw_rate=0.003 net_yaw_ratio=0.941 wz_ratio=0.945 yaw_ripple=0.034 heading_drift=n/a planar_drift=0.010 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  turn_w_max: net_yaw_rate=-0.002 net_yaw_ratio=0.872 wz_ratio=0.875 yaw_ripple=0.026 heading_drift=n/a planar_drift=0.013 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
