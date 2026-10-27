```
=== crab-hex gait eval ===
scenario   : turnlevels__omni_body  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_28000.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7616366028142711  p25=0.734718382202491  p75=0.7837280794815066
tippy_tap_fraction median=0.4225352112676056
slip_ratio         median=0.25371724799439754
tracking_ratio     median=None  (achieved/commanded vx, walking holds; n=0)
schedule_completion_rate=1.0
terminations={'schedule_complete': 1000}
facing_error_rms   median=0.20079984688338326  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.6736843321711292 (n=926)
  turn_l_w_min: cmd 0.00 -> achieved -0.001 m/s | tripod median=0.7610717375092887 (n=100)
  turn_r_w_min: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.7436839929711798 (n=100)
  turn_l_0p08: cmd 0.00 -> achieved -0.002 m/s | tripod median=0.7625742321162399 (n=100)
  turn_r_0p08: cmd 0.00 -> achieved 0.005 m/s | tripod median=0.7528838986319334 (n=100)
  turn_l_0p15: cmd 0.00 -> achieved -0.003 m/s | tripod median=0.7627758968626825 (n=100)
  turn_r_0p15: cmd 0.00 -> achieved 0.007 m/s | tripod median=0.7841993670314666 (n=100)
  turn_l_ceiling: cmd 0.00 -> achieved -0.002 m/s | tripod median=0.7770315910920036 (n=100)
  turn_r_ceiling: cmd 0.00 -> achieved 0.007 m/s | tripod median=0.7929070657820307 (n=100)
  turn_l_w_max: cmd 0.00 -> achieved -0.006 m/s | tripod median=0.7798282061693331 (n=100)
  turn_r_w_max: cmd 0.00 -> achieved 0.007 m/s | tripod median=0.8053922102441099 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.014 initial=0.009 final=0.015 max=0.030 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.000 yaw_rate_ratio=n/a
  turn_l_w_min: err rms=0.116 initial=0.017 final=0.128 max=0.194 closed=n/a | settle=11.240 (n=73/100) | progress=0.048 net=0.010 yaw_rate_ratio=0.577
  turn_r_w_min: err rms=0.126 initial=0.016 final=0.126 max=0.216 closed=n/a | settle=11.260 (n=75/100) | progress=0.049 net=-0.007 yaw_rate_ratio=0.518
  turn_l_0p08: err rms=0.136 initial=0.012 final=0.149 max=0.224 closed=n/a | settle=11.660 (n=51/100) | progress=0.077 net=0.013 yaw_rate_ratio=0.628
  turn_r_0p08: err rms=0.149 initial=0.012 final=0.150 max=0.243 closed=n/a | settle=11.530 (n=48/100) | progress=0.078 net=-0.009 yaw_rate_ratio=0.544
  turn_l_0p15: err rms=0.230 initial=0.013 final=0.261 max=0.341 closed=n/a | settle=11.770 (n=2/100) | progress=0.139 net=0.014 yaw_rate_ratio=0.597
  turn_r_0p15: err rms=0.259 initial=0.017 final=0.307 max=0.393 closed=n/a | settle=n/a (n=0/100) | progress=0.135 net=-0.020 yaw_rate_ratio=0.594
  turn_l_ceiling: err rms=0.254 initial=0.015 final=0.285 max=0.355 closed=n/a | settle=n/a (n=0/100) | progress=0.155 net=0.020 yaw_rate_ratio=0.620
  turn_r_ceiling: err rms=0.294 initial=0.016 final=0.373 max=0.439 closed=n/a | settle=n/a (n=0/100) | progress=0.147 net=-0.025 yaw_rate_ratio=0.614
  turn_l_w_max: err rms=0.386 initial=0.014 final=0.551 max=0.568 closed=n/a | settle=n/a (n=0/100) | progress=0.173 net=0.044 yaw_rate_ratio=0.648
  turn_r_w_max: err rms=0.404 initial=0.019 final=0.542 max=0.575 closed=n/a | settle=n/a (n=0/100) | progress=0.174 net=-0.041 yaw_rate_ratio=0.643

body yaw by hold (window [4 s, hold_s); medians; net = net yaw rate rad/s, ratio = net / wz_cmd SIGNED (the gated turn number), wz_ratio = mean wz / wz_cmd; ripple = std of the 1-s mean wz; drift = |mean wz| on wz = 0 holds; planar = xy drift m/s on pure turns; body = along-bearing speed / |v_cmd|, cross = mean |perp| m/s):
          settle: net=n/a ratio=n/a wz_ratio=n/a ripple=n/a | drift=n/a planar=n/a | body=n/a cross=n/a (n=0)
    turn_l_w_min: net=0.049 ratio=0.974 wz_ratio=0.972 ripple=0.033 | drift=n/a planar=0.010 | body=n/a cross=n/a (n=100)
    turn_r_w_min: net=-0.052 ratio=1.043 wz_ratio=1.041 ripple=0.040 | drift=n/a planar=0.006 | body=n/a cross=n/a (n=100)
     turn_l_0p08: net=0.078 ratio=0.979 wz_ratio=0.983 ripple=0.037 | drift=n/a planar=0.010 | body=n/a cross=n/a (n=100)
     turn_r_0p08: net=-0.082 ratio=1.028 wz_ratio=1.029 ripple=0.039 | drift=n/a planar=0.006 | body=n/a cross=n/a (n=100)
     turn_l_0p15: net=0.145 ratio=0.967 wz_ratio=0.974 ripple=0.033 | drift=n/a planar=0.010 | body=n/a cross=n/a (n=100)
     turn_r_0p15: net=-0.140 ratio=0.932 wz_ratio=0.934 ripple=0.049 | drift=n/a planar=0.010 | body=n/a cross=n/a (n=100)
  turn_l_ceiling: net=0.162 ratio=0.963 wz_ratio=0.970 ripple=0.027 | drift=n/a planar=0.010 | body=n/a cross=n/a (n=100)
  turn_r_ceiling: net=-0.154 ratio=0.918 wz_ratio=0.922 ripple=0.046 | drift=n/a planar=0.010 | body=n/a cross=n/a (n=100)
    turn_l_w_max: net=0.182 ratio=0.869 wz_ratio=0.873 ripple=0.021 | drift=n/a planar=0.013 | body=n/a cross=n/a (n=100)
    turn_r_w_max: net=-0.185 ratio=0.883 wz_ratio=0.886 ripple=0.030 | drift=n/a planar=0.011 | body=n/a cross=n/a (n=100)

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: n/a):
  turn_w_min: n=200 surv=1.000 (turn_l_w_min=1.00 turn_r_w_min=1.00) | ratio=n/a achieved=0.001 tripod=0.754 | closed=n/a progress=0.048 net=-0.004 | drift=0.040 dir_err=n/a
  turn_0p08: n=200 surv=1.000 (turn_l_0p08=1.00 turn_r_0p08=1.00) | ratio=n/a achieved=0.001 tripod=0.759 | closed=n/a progress=0.077 net=0.001 | drift=0.042 dir_err=n/a
  turn_0p15: n=200 surv=1.000 (turn_l_0p15=1.00 turn_r_0p15=1.00) | ratio=n/a achieved=0.002 tripod=0.772 | closed=n/a progress=0.136 net=-0.016 | drift=0.043 dir_err=n/a
  turn_ceiling: n=200 surv=1.000 (turn_l_ceiling=1.00 turn_r_ceiling=1.00) | ratio=n/a achieved=0.003 tripod=0.783 | closed=n/a progress=0.151 net=-0.020 | drift=0.043 dir_err=n/a
  turn_w_max: n=200 surv=1.000 (turn_l_w_max=1.00 turn_r_w_max=1.00) | ratio=n/a achieved=0.002 tripod=0.794 | closed=n/a progress=0.174 net=-0.038 | drift=0.047 dir_err=n/a

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
  turn_w_min: net_yaw_rate=-0.001 net_yaw_ratio=1.011 wz_ratio=1.007 yaw_ripple=0.035 heading_drift=n/a planar_drift=0.008 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  turn_0p08: net_yaw_rate=-0.002 net_yaw_ratio=1.005 wz_ratio=1.007 yaw_ripple=0.038 heading_drift=n/a planar_drift=0.009 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  turn_0p15: net_yaw_rate=-0.002 net_yaw_ratio=0.948 wz_ratio=0.951 yaw_ripple=0.040 heading_drift=n/a planar_drift=0.010 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  turn_ceiling: net_yaw_rate=0.007 net_yaw_ratio=0.936 wz_ratio=0.939 yaw_ripple=0.035 heading_drift=n/a planar_drift=0.010 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  turn_w_max: net_yaw_rate=0.011 net_yaw_ratio=0.874 wz_ratio=0.876 yaw_ripple=0.025 heading_drift=n/a planar_drift=0.012 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
