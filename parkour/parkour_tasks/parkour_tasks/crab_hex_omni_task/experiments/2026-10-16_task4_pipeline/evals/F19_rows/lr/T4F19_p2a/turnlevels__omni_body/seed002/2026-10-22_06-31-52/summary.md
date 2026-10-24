```
=== crab-hex gait eval ===
scenario   : turnlevels__omni_body  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-21_23-06-12_seed19_X06_p2a_A15pB/model_19994.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7800180587883896  p25=0.7582887487115794  p75=0.7973404259183731
tippy_tap_fraction median=0.4039205344585092
slip_ratio         median=0.21259253145672785
tracking_ratio     median=None  (achieved/commanded vx, walking holds; n=0)
schedule_completion_rate=1.0
terminations={'schedule_complete': 1000}
facing_error_rms   median=0.25345929042161774  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.7795719161703892 (n=983)
  turn_l_w_min: cmd 0.00 -> achieved -0.006 m/s | tripod median=0.7470028040527746 (n=100)
  turn_r_w_min: cmd 0.00 -> achieved 0.006 m/s | tripod median=0.7761377412452266 (n=100)
  turn_l_0p08: cmd 0.00 -> achieved -0.005 m/s | tripod median=0.7546224191678922 (n=100)
  turn_r_0p08: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.7729919572209314 (n=100)
  turn_l_0p15: cmd 0.00 -> achieved -0.000 m/s | tripod median=0.7816909502076818 (n=100)
  turn_r_0p15: cmd 0.00 -> achieved -0.003 m/s | tripod median=0.7930802972658739 (n=100)
  turn_l_ceiling: cmd 0.00 -> achieved -0.001 m/s | tripod median=0.780874354512471 (n=100)
  turn_r_ceiling: cmd 0.00 -> achieved -0.002 m/s | tripod median=0.7920132542310536 (n=100)
  turn_l_w_max: cmd 0.00 -> achieved -0.000 m/s | tripod median=0.7909851237831202 (n=100)
  turn_r_w_max: cmd 0.00 -> achieved -0.002 m/s | tripod median=0.803202230431034 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.019 initial=0.006 final=0.015 max=0.035 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.001 yaw_rate_ratio=n/a
  turn_l_w_min: err rms=0.120 initial=0.013 final=0.145 max=0.194 closed=n/a | settle=11.640 (n=59/100) | progress=0.048 net=0.012 yaw_rate_ratio=0.552
  turn_r_w_min: err rms=0.133 initial=0.016 final=0.159 max=0.210 closed=n/a | settle=11.570 (n=40/100) | progress=0.047 net=-0.009 yaw_rate_ratio=0.527
  turn_l_0p08: err rms=0.152 initial=0.014 final=0.155 max=0.239 closed=n/a | settle=11.800 (n=41/100) | progress=0.078 net=0.013 yaw_rate_ratio=0.598
  turn_r_0p08: err rms=0.168 initial=0.017 final=0.199 max=0.264 closed=n/a | settle=11.720 (n=7/100) | progress=0.074 net=-0.013 yaw_rate_ratio=0.541
  turn_l_0p15: err rms=0.267 initial=0.014 final=0.349 max=0.395 closed=n/a | settle=n/a (n=0/100) | progress=0.131 net=0.027 yaw_rate_ratio=0.585
  turn_r_0p15: err rms=0.316 initial=0.019 final=0.410 max=0.466 closed=n/a | settle=n/a (n=0/100) | progress=0.126 net=-0.031 yaw_rate_ratio=0.557
  turn_l_ceiling: err rms=0.322 initial=0.017 final=0.430 max=0.473 closed=n/a | settle=n/a (n=0/100) | progress=0.142 net=0.032 yaw_rate_ratio=0.589
  turn_r_ceiling: err rms=0.355 initial=0.017 final=0.484 max=0.527 closed=n/a | settle=n/a (n=0/100) | progress=0.137 net=-0.036 yaw_rate_ratio=0.570
  turn_l_w_max: err rms=0.455 initial=0.014 final=0.568 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.157 net=0.047 yaw_rate_ratio=0.588
  turn_r_w_max: err rms=0.480 initial=0.016 final=0.579 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.146 net=-0.046 yaw_rate_ratio=0.564

body yaw by hold (window [4 s, hold_s); medians; net = net yaw rate rad/s, ratio = net / wz_cmd SIGNED (the gated turn number), wz_ratio = mean wz / wz_cmd; ripple = std of the 1-s mean wz; drift = |mean wz| on wz = 0 holds; planar = xy drift m/s on pure turns; body = along-bearing speed / |v_cmd|, cross = mean |perp| m/s):
          settle: net=n/a ratio=n/a wz_ratio=n/a ripple=n/a | drift=n/a planar=n/a | body=n/a cross=n/a (n=0)
    turn_l_w_min: net=0.047 ratio=0.932 wz_ratio=0.939 ripple=0.027 | drift=n/a planar=0.010 | body=n/a cross=n/a (n=100)
    turn_r_w_min: net=-0.047 ratio=0.949 wz_ratio=0.959 ripple=0.028 | drift=n/a planar=0.007 | body=n/a cross=n/a (n=100)
     turn_l_0p08: net=0.080 ratio=0.996 wz_ratio=1.005 ripple=0.030 | drift=n/a planar=0.007 | body=n/a cross=n/a (n=100)
     turn_r_0p08: net=-0.077 ratio=0.966 wz_ratio=0.976 ripple=0.030 | drift=n/a planar=0.005 | body=n/a cross=n/a (n=100)
     turn_l_0p15: net=0.135 ratio=0.903 wz_ratio=0.908 ripple=0.031 | drift=n/a planar=0.006 | body=n/a cross=n/a (n=100)
     turn_r_0p15: net=-0.133 ratio=0.887 wz_ratio=0.893 ripple=0.026 | drift=n/a planar=0.008 | body=n/a cross=n/a (n=100)
  turn_l_ceiling: net=0.148 ratio=0.882 wz_ratio=0.886 ripple=0.029 | drift=n/a planar=0.007 | body=n/a cross=n/a (n=100)
  turn_r_ceiling: net=-0.146 ratio=0.867 wz_ratio=0.870 ripple=0.026 | drift=n/a planar=0.009 | body=n/a cross=n/a (n=100)
    turn_l_w_max: net=0.161 ratio=0.769 wz_ratio=0.772 ripple=0.025 | drift=n/a planar=0.006 | body=n/a cross=n/a (n=100)
    turn_r_w_max: net=-0.154 ratio=0.735 wz_ratio=0.739 ripple=0.025 | drift=n/a planar=0.012 | body=n/a cross=n/a (n=100)

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: n/a):
  turn_w_min: n=200 surv=1.000 (turn_l_w_min=1.00 turn_r_w_min=1.00) | ratio=n/a achieved=0.000 tripod=0.765 | closed=n/a progress=0.048 net=-0.005 | drift=0.043 dir_err=n/a
  turn_0p08: n=200 surv=1.000 (turn_l_0p08=1.00 turn_r_0p08=1.00) | ratio=n/a achieved=-0.002 tripod=0.761 | closed=n/a progress=0.076 net=-0.009 | drift=0.044 dir_err=n/a
  turn_0p15: n=200 surv=1.000 (turn_l_0p15=1.00 turn_r_0p15=1.00) | ratio=n/a achieved=-0.002 tripod=0.785 | closed=n/a progress=0.128 net=-0.026 | drift=0.045 dir_err=n/a
  turn_ceiling: n=200 surv=1.000 (turn_l_ceiling=1.00 turn_r_ceiling=1.00) | ratio=n/a achieved=-0.001 tripod=0.786 | closed=n/a progress=0.140 net=-0.032 | drift=0.046 dir_err=n/a
  turn_w_max: n=200 surv=1.000 (turn_l_w_max=1.00 turn_r_w_max=1.00) | ratio=n/a achieved=-0.001 tripod=0.797 | closed=n/a progress=0.151 net=-0.044 | drift=0.049 dir_err=n/a

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
  turn_w_min: net_yaw_rate=-0.000 net_yaw_ratio=0.940 wz_ratio=0.952 yaw_ripple=0.027 heading_drift=n/a planar_drift=0.008 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  turn_0p08: net_yaw_rate=0.003 net_yaw_ratio=0.983 wz_ratio=0.991 yaw_ripple=0.030 heading_drift=n/a planar_drift=0.006 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  turn_0p15: net_yaw_rate=0.003 net_yaw_ratio=0.894 wz_ratio=0.902 yaw_ripple=0.028 heading_drift=n/a planar_drift=0.007 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  turn_ceiling: net_yaw_rate=0.001 net_yaw_ratio=0.871 wz_ratio=0.876 yaw_ripple=0.027 heading_drift=n/a planar_drift=0.007 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  turn_w_max: net_yaw_rate=0.009 net_yaw_ratio=0.751 wz_ratio=0.755 yaw_ripple=0.025 heading_drift=n/a planar_drift=0.009 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
