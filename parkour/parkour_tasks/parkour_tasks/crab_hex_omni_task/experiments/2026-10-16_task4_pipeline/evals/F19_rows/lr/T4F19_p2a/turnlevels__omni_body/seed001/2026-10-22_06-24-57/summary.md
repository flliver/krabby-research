```
=== crab-hex gait eval ===
scenario   : turnlevels__omni_body  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-21_23-06-12_seed19_X06_p2a_A15pB/model_19994.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7819751954981096  p25=0.7593961137430993  p75=0.799689490535501
tippy_tap_fraction median=0.40540540540540543
slip_ratio         median=0.21362271999626098
tracking_ratio     median=None  (achieved/commanded vx, walking holds; n=0)
schedule_completion_rate=1.0
terminations={'schedule_complete': 1000}
facing_error_rms   median=0.2534405485036758  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7758821897059103 (n=974)
  turn_l_w_min: cmd 0.00 -> achieved -0.006 m/s | tripod median=0.7579140448069988 (n=100)
  turn_r_w_min: cmd 0.00 -> achieved 0.005 m/s | tripod median=0.7754212645734396 (n=100)
  turn_l_0p08: cmd 0.00 -> achieved -0.005 m/s | tripod median=0.750272665738567 (n=100)
  turn_r_0p08: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.7721184904536257 (n=100)
  turn_l_0p15: cmd 0.00 -> achieved -0.000 m/s | tripod median=0.7841818766445647 (n=100)
  turn_r_0p15: cmd 0.00 -> achieved -0.001 m/s | tripod median=0.7939916128865161 (n=100)
  turn_l_ceiling: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.7871986590014848 (n=100)
  turn_r_ceiling: cmd 0.00 -> achieved -0.001 m/s | tripod median=0.79345271025401 (n=100)
  turn_l_w_max: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.7977909034749764 (n=100)
  turn_r_w_max: cmd 0.00 -> achieved -0.001 m/s | tripod median=0.8024370267321151 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.017 initial=0.005 final=0.014 max=0.033 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.002 yaw_rate_ratio=n/a
  turn_l_w_min: err rms=0.121 initial=0.015 final=0.143 max=0.192 closed=n/a | settle=11.580 (n=58/100) | progress=0.048 net=0.012 yaw_rate_ratio=0.569
  turn_r_w_min: err rms=0.130 initial=0.017 final=0.163 max=0.210 closed=n/a | settle=11.580 (n=39/100) | progress=0.046 net=-0.010 yaw_rate_ratio=0.525
  turn_l_0p08: err rms=0.153 initial=0.014 final=0.160 max=0.238 closed=n/a | settle=11.780 (n=39/100) | progress=0.077 net=0.012 yaw_rate_ratio=0.600
  turn_r_0p08: err rms=0.168 initial=0.017 final=0.203 max=0.262 closed=n/a | settle=11.640 (n=6/100) | progress=0.073 net=-0.014 yaw_rate_ratio=0.535
  turn_l_0p15: err rms=0.276 initial=0.014 final=0.339 max=0.397 closed=n/a | settle=n/a (n=0/100) | progress=0.132 net=-0.020 yaw_rate_ratio=0.582
  turn_r_0p15: err rms=0.311 initial=0.014 final=0.399 max=0.458 closed=n/a | settle=n/a (n=0/100) | progress=0.127 net=-0.029 yaw_rate_ratio=0.558
  turn_l_ceiling: err rms=0.322 initial=0.014 final=0.425 max=0.465 closed=n/a | settle=n/a (n=0/100) | progress=0.142 net=0.030 yaw_rate_ratio=0.588
  turn_r_ceiling: err rms=0.367 initial=0.015 final=0.480 max=0.524 closed=n/a | settle=n/a (n=0/100) | progress=0.137 net=-0.037 yaw_rate_ratio=0.565
  turn_l_w_max: err rms=0.455 initial=0.011 final=0.568 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.157 net=0.046 yaw_rate_ratio=0.591
  turn_r_w_max: err rms=0.480 initial=0.014 final=0.582 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.147 net=-0.046 yaw_rate_ratio=0.559

body yaw by hold (window [4 s, hold_s); medians; net = net yaw rate rad/s, ratio = net / wz_cmd SIGNED (the gated turn number), wz_ratio = mean wz / wz_cmd; ripple = std of the 1-s mean wz; drift = |mean wz| on wz = 0 holds; planar = xy drift m/s on pure turns; body = along-bearing speed / |v_cmd|, cross = mean |perp| m/s):
          settle: net=n/a ratio=n/a wz_ratio=n/a ripple=n/a | drift=n/a planar=n/a | body=n/a cross=n/a (n=0)
    turn_l_w_min: net=0.047 ratio=0.950 wz_ratio=0.952 ripple=0.023 | drift=n/a planar=0.009 | body=n/a cross=n/a (n=100)
    turn_r_w_min: net=-0.047 ratio=0.944 wz_ratio=0.949 ripple=0.025 | drift=n/a planar=0.007 | body=n/a cross=n/a (n=100)
     turn_l_0p08: net=0.080 ratio=1.006 wz_ratio=1.015 ripple=0.031 | drift=n/a planar=0.007 | body=n/a cross=n/a (n=100)
     turn_r_0p08: net=-0.077 ratio=0.961 wz_ratio=0.970 ripple=0.029 | drift=n/a planar=0.004 | body=n/a cross=n/a (n=100)
     turn_l_0p15: net=0.137 ratio=0.916 wz_ratio=0.921 ripple=0.031 | drift=n/a planar=0.006 | body=n/a cross=n/a (n=100)
     turn_r_0p15: net=-0.134 ratio=0.895 wz_ratio=0.899 ripple=0.027 | drift=n/a planar=0.008 | body=n/a cross=n/a (n=100)
  turn_l_ceiling: net=0.148 ratio=0.884 wz_ratio=0.889 ripple=0.026 | drift=n/a planar=0.007 | body=n/a cross=n/a (n=100)
  turn_r_ceiling: net=-0.145 ratio=0.861 wz_ratio=0.869 ripple=0.024 | drift=n/a planar=0.009 | body=n/a cross=n/a (n=100)
    turn_l_w_max: net=0.162 ratio=0.769 wz_ratio=0.773 ripple=0.025 | drift=n/a planar=0.007 | body=n/a cross=n/a (n=100)
    turn_r_w_max: net=-0.154 ratio=0.731 wz_ratio=0.736 ripple=0.025 | drift=n/a planar=0.012 | body=n/a cross=n/a (n=100)

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: n/a):
  turn_w_min: n=200 surv=1.000 (turn_l_w_min=1.00 turn_r_w_min=1.00) | ratio=n/a achieved=-0.001 tripod=0.767 | closed=n/a progress=0.046 net=-0.005 | drift=0.043 dir_err=n/a
  turn_0p08: n=200 surv=1.000 (turn_l_0p08=1.00 turn_r_0p08=1.00) | ratio=n/a achieved=-0.002 tripod=0.762 | closed=n/a progress=0.075 net=-0.009 | drift=0.043 dir_err=n/a
  turn_0p15: n=200 surv=1.000 (turn_l_0p15=1.00 turn_r_0p15=1.00) | ratio=n/a achieved=-0.001 tripod=0.788 | closed=n/a progress=0.129 net=-0.025 | drift=0.044 dir_err=n/a
  turn_ceiling: n=200 surv=1.000 (turn_l_ceiling=1.00 turn_r_ceiling=1.00) | ratio=n/a achieved=-0.001 tripod=0.789 | closed=n/a progress=0.140 net=-0.033 | drift=0.046 dir_err=n/a
  turn_w_max: n=200 surv=1.000 (turn_l_w_max=1.00 turn_r_w_max=1.00) | ratio=n/a achieved=-0.000 tripod=0.801 | closed=n/a progress=0.152 net=-0.045 | drift=0.048 dir_err=n/a

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
  turn_w_min: net_yaw_rate=-0.001 net_yaw_ratio=0.949 wz_ratio=0.951 yaw_ripple=0.024 heading_drift=n/a planar_drift=0.008 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  turn_0p08: net_yaw_rate=0.004 net_yaw_ratio=0.985 wz_ratio=0.997 yaw_ripple=0.030 heading_drift=n/a planar_drift=0.006 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  turn_0p15: net_yaw_rate=0.003 net_yaw_ratio=0.902 wz_ratio=0.907 yaw_ripple=0.029 heading_drift=n/a planar_drift=0.007 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  turn_ceiling: net_yaw_rate=0.004 net_yaw_ratio=0.877 wz_ratio=0.882 yaw_ripple=0.025 heading_drift=n/a planar_drift=0.007 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  turn_w_max: net_yaw_rate=0.007 net_yaw_ratio=0.752 wz_ratio=0.756 yaw_ripple=0.025 heading_drift=n/a planar_drift=0.009 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
