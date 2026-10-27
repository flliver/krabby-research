```
=== crab-hex gait eval ===
scenario   : turnlevels__omni_body  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7787520578458094  p25=0.761009383036647  p75=0.796506782338801
tippy_tap_fraction median=0.44871794871794873
slip_ratio         median=0.2507451121081423
tracking_ratio     median=None  (achieved/commanded vx, walking holds; n=0)
schedule_completion_rate=1.0
terminations={'schedule_complete': 1000}
facing_error_rms   median=0.17419576441477402  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.7714153275434491 (n=978)
  turn_l_w_min: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.7752691225636748 (n=100)
  turn_r_w_min: cmd 0.00 -> achieved 0.005 m/s | tripod median=0.7638125565296723 (n=100)
  turn_l_0p08: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.778375453784885 (n=100)
  turn_r_0p08: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7609781254452479 (n=100)
  turn_l_0p15: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.7764511092928126 (n=100)
  turn_r_0p15: cmd 0.00 -> achieved 0.006 m/s | tripod median=0.7854169179305952 (n=100)
  turn_l_ceiling: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.7862049281123125 (n=100)
  turn_r_ceiling: cmd 0.00 -> achieved 0.005 m/s | tripod median=0.7889424011906814 (n=100)
  turn_l_w_max: cmd 0.00 -> achieved -0.004 m/s | tripod median=0.7849035317482886 (n=100)
  turn_r_w_max: cmd 0.00 -> achieved 0.007 m/s | tripod median=0.8041219108612818 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.014 initial=0.008 final=0.012 max=0.028 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.001 yaw_rate_ratio=n/a
  turn_l_w_min: err rms=0.108 initial=0.014 final=0.117 max=0.180 closed=n/a | settle=11.120 (n=86/100) | progress=0.049 net=-0.006 yaw_rate_ratio=0.646
  turn_r_w_min: err rms=0.117 initial=0.013 final=0.127 max=0.190 closed=n/a | settle=11.360 (n=78/100) | progress=0.049 net=-0.007 yaw_rate_ratio=0.529
  turn_l_0p08: err rms=0.130 initial=0.012 final=0.142 max=0.208 closed=n/a | settle=11.550 (n=60/100) | progress=0.078 net=0.008 yaw_rate_ratio=0.669
  turn_r_0p08: err rms=0.137 initial=0.012 final=0.138 max=0.223 closed=n/a | settle=11.620 (n=63/100) | progress=0.078 net=-0.008 yaw_rate_ratio=0.566
  turn_l_0p15: err rms=0.183 initial=0.013 final=0.195 max=0.275 closed=n/a | settle=11.860 (n=23/100) | progress=0.145 net=-0.010 yaw_rate_ratio=0.645
  turn_r_0p15: err rms=0.235 initial=0.014 final=0.293 max=0.358 closed=n/a | settle=n/a (n=0/100) | progress=0.136 net=-0.018 yaw_rate_ratio=0.604
  turn_l_ceiling: err rms=0.210 initial=0.009 final=0.227 max=0.293 closed=n/a | settle=11.920 (n=3/100) | progress=0.160 net=0.015 yaw_rate_ratio=0.633
  turn_r_ceiling: err rms=0.279 initial=0.015 final=0.339 max=0.416 closed=n/a | settle=n/a (n=0/100) | progress=0.150 net=-0.023 yaw_rate_ratio=0.615
  turn_l_w_max: err rms=0.364 initial=0.012 final=0.508 max=0.528 closed=n/a | settle=n/a (n=0/100) | progress=0.178 net=-0.036 yaw_rate_ratio=0.655
  turn_r_w_max: err rms=0.395 initial=0.010 final=0.543 max=0.588 closed=n/a | settle=n/a (n=0/100) | progress=0.174 net=-0.041 yaw_rate_ratio=0.643

body yaw by hold (window [4 s, hold_s); medians; net = net yaw rate rad/s, ratio = net / wz_cmd SIGNED (the gated turn number), wz_ratio = mean wz / wz_cmd; ripple = std of the 1-s mean wz; drift = |mean wz| on wz = 0 holds; planar = xy drift m/s on pure turns; body = along-bearing speed / |v_cmd|, cross = mean |perp| m/s):
          settle: net=n/a ratio=n/a wz_ratio=n/a ripple=n/a | drift=n/a planar=n/a | body=n/a cross=n/a (n=0)
    turn_l_w_min: net=0.050 ratio=1.002 wz_ratio=1.011 ripple=0.027 | drift=n/a planar=0.004 | body=n/a cross=n/a (n=100)
    turn_r_w_min: net=-0.050 ratio=0.991 wz_ratio=0.992 ripple=0.032 | drift=n/a planar=0.007 | body=n/a cross=n/a (n=100)
     turn_l_0p08: net=0.079 ratio=0.992 wz_ratio=0.997 ripple=0.029 | drift=n/a planar=0.004 | body=n/a cross=n/a (n=100)
     turn_r_0p08: net=-0.082 ratio=1.022 wz_ratio=1.028 ripple=0.034 | drift=n/a planar=0.007 | body=n/a cross=n/a (n=100)
     turn_l_0p15: net=0.149 ratio=0.990 wz_ratio=0.996 ripple=0.027 | drift=n/a planar=0.005 | body=n/a cross=n/a (n=100)
     turn_r_0p15: net=-0.138 ratio=0.917 wz_ratio=0.922 ripple=0.048 | drift=n/a planar=0.008 | body=n/a cross=n/a (n=100)
  turn_l_ceiling: net=0.167 ratio=0.991 wz_ratio=0.996 ripple=0.023 | drift=n/a planar=0.005 | body=n/a cross=n/a (n=100)
  turn_r_ceiling: net=-0.152 ratio=0.908 wz_ratio=0.911 ripple=0.042 | drift=n/a planar=0.008 | body=n/a cross=n/a (n=100)
    turn_l_w_max: net=0.184 ratio=0.878 wz_ratio=0.883 ripple=0.023 | drift=n/a planar=0.009 | body=n/a cross=n/a (n=100)
    turn_r_w_max: net=-0.182 ratio=0.865 wz_ratio=0.868 ripple=0.030 | drift=n/a planar=0.010 | body=n/a cross=n/a (n=100)

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: n/a):
  turn_w_min: n=200 surv=1.000 (turn_l_w_min=1.00 turn_r_w_min=1.00) | ratio=n/a achieved=0.003 tripod=0.770 | closed=n/a progress=0.049 net=-0.006 | drift=0.038 dir_err=n/a
  turn_0p08: n=200 surv=1.000 (turn_l_0p08=1.00 turn_r_0p08=1.00) | ratio=n/a achieved=0.002 tripod=0.772 | closed=n/a progress=0.078 net=-0.007 | drift=0.039 dir_err=n/a
  turn_0p15: n=200 surv=1.000 (turn_l_0p15=1.00 turn_r_0p15=1.00) | ratio=n/a achieved=0.003 tripod=0.780 | closed=n/a progress=0.141 net=-0.014 | drift=0.042 dir_err=n/a
  turn_ceiling: n=200 surv=1.000 (turn_l_ceiling=1.00 turn_r_ceiling=1.00) | ratio=n/a achieved=0.003 tripod=0.787 | closed=n/a progress=0.156 net=-0.015 | drift=0.042 dir_err=n/a
  turn_w_max: n=200 surv=1.000 (turn_l_w_max=1.00 turn_r_w_max=1.00) | ratio=n/a achieved=0.001 tripod=0.796 | closed=n/a progress=0.176 net=-0.038 | drift=0.046 dir_err=n/a

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
  turn_w_min: net_yaw_rate=0.001 net_yaw_ratio=0.997 wz_ratio=1.001 yaw_ripple=0.030 heading_drift=n/a planar_drift=0.005 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  turn_0p08: net_yaw_rate=0.000 net_yaw_ratio=1.007 wz_ratio=1.011 yaw_ripple=0.033 heading_drift=n/a planar_drift=0.005 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  turn_0p15: net_yaw_rate=0.004 net_yaw_ratio=0.962 wz_ratio=0.968 yaw_ripple=0.036 heading_drift=n/a planar_drift=0.006 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  turn_ceiling: net_yaw_rate=0.009 net_yaw_ratio=0.959 wz_ratio=0.965 yaw_ripple=0.031 heading_drift=n/a planar_drift=0.006 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  turn_w_max: net_yaw_rate=0.006 net_yaw_ratio=0.871 wz_ratio=0.876 yaw_ripple=0.026 heading_drift=n/a planar_drift=0.010 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
