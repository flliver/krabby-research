```
=== crab-hex gait eval ===
scenario   : turnlevels__omni_body  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt
episodes   : 1000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7797786051672406  p25=0.7612029068904672  p75=0.7945276104412804
tippy_tap_fraction median=0.4523809523809524
slip_ratio         median=0.24345771320094156
tracking_ratio     median=None  (achieved/commanded vx, walking holds; n=0)
schedule_completion_rate=1.0
terminations={'schedule_complete': 1000}
facing_error_rms   median=0.17317993990759833  (rad, whole episode; n=1000)

by hold:
    settle: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.766559066826343 (n=977)
  turn_l_w_min: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.7771377512662248 (n=100)
  turn_r_w_min: cmd 0.00 -> achieved 0.005 m/s | tripod median=0.7576271832971386 (n=100)
  turn_l_0p08: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.7825944076969739 (n=100)
  turn_r_0p08: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7653837125569334 (n=100)
  turn_l_0p15: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.7855639864891077 (n=100)
  turn_r_0p15: cmd 0.00 -> achieved 0.005 m/s | tripod median=0.785901942726767 (n=100)
  turn_l_ceiling: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.7821949507303346 (n=100)
  turn_r_ceiling: cmd 0.00 -> achieved 0.005 m/s | tripod median=0.7839378787001864 (n=100)
  turn_l_w_max: cmd 0.00 -> achieved -0.005 m/s | tripod median=0.7894445304654937 (n=100)
  turn_r_w_max: cmd 0.00 -> achieved 0.007 m/s | tripod median=0.8012395302140808 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.013 initial=0.008 final=0.013 max=0.028 closed=n/a | settle=0.000 (n=1000/1000) | progress=n/a net=0.002 yaw_rate_ratio=n/a
  turn_l_w_min: err rms=0.108 initial=0.012 final=0.115 max=0.180 closed=n/a | settle=11.060 (n=85/100) | progress=0.048 net=-0.005 yaw_rate_ratio=0.644
  turn_r_w_min: err rms=0.118 initial=0.012 final=0.124 max=0.192 closed=n/a | settle=10.960 (n=76/100) | progress=0.049 net=-0.003 yaw_rate_ratio=0.543
  turn_l_0p08: err rms=0.129 initial=0.013 final=0.139 max=0.205 closed=n/a | settle=11.600 (n=64/100) | progress=0.079 net=0.008 yaw_rate_ratio=0.675
  turn_r_0p08: err rms=0.138 initial=0.014 final=0.152 max=0.224 closed=n/a | settle=11.600 (n=47/100) | progress=0.077 net=-0.008 yaw_rate_ratio=0.564
  turn_l_0p15: err rms=0.187 initial=0.013 final=0.198 max=0.273 closed=n/a | settle=11.740 (n=18/100) | progress=0.144 net=-0.010 yaw_rate_ratio=0.638
  turn_r_0p15: err rms=0.237 initial=0.013 final=0.269 max=0.361 closed=n/a | settle=n/a (n=0/100) | progress=0.138 net=-0.017 yaw_rate_ratio=0.604
  turn_l_ceiling: err rms=0.211 initial=0.013 final=0.230 max=0.291 closed=n/a | settle=11.880 (n=1/100) | progress=0.160 net=-0.013 yaw_rate_ratio=0.632
  turn_r_ceiling: err rms=0.278 initial=0.012 final=0.348 max=0.417 closed=n/a | settle=n/a (n=0/100) | progress=0.149 net=-0.023 yaw_rate_ratio=0.620
  turn_l_w_max: err rms=0.364 initial=0.010 final=0.505 max=0.533 closed=n/a | settle=n/a (n=0/100) | progress=0.178 net=-0.036 yaw_rate_ratio=0.655
  turn_r_w_max: err rms=0.397 initial=0.015 final=0.549 max=0.595 closed=n/a | settle=n/a (n=0/100) | progress=0.173 net=-0.042 yaw_rate_ratio=0.644

body yaw by hold (window [4 s, hold_s); medians; net = net yaw rate rad/s, ratio = net / wz_cmd SIGNED (the gated turn number), wz_ratio = mean wz / wz_cmd; ripple = std of the 1-s mean wz; drift = |mean wz| on wz = 0 holds; planar = xy drift m/s on pure turns; body = along-bearing speed / |v_cmd|, cross = mean |perp| m/s):
          settle: net=n/a ratio=n/a wz_ratio=n/a ripple=n/a | drift=n/a planar=n/a | body=n/a cross=n/a (n=0)
    turn_l_w_min: net=0.050 ratio=0.996 wz_ratio=1.001 ripple=0.028 | drift=n/a planar=0.004 | body=n/a cross=n/a (n=100)
    turn_r_w_min: net=-0.051 ratio=1.011 wz_ratio=1.006 ripple=0.034 | drift=n/a planar=0.007 | body=n/a cross=n/a (n=100)
     turn_l_0p08: net=0.079 ratio=0.991 wz_ratio=0.999 ripple=0.030 | drift=n/a planar=0.004 | body=n/a cross=n/a (n=100)
     turn_r_0p08: net=-0.080 ratio=0.997 wz_ratio=1.003 ripple=0.038 | drift=n/a planar=0.006 | body=n/a cross=n/a (n=100)
     turn_l_0p15: net=0.148 ratio=0.987 wz_ratio=0.993 ripple=0.031 | drift=n/a planar=0.005 | body=n/a cross=n/a (n=100)
     turn_r_0p15: net=-0.140 ratio=0.936 wz_ratio=0.938 ripple=0.046 | drift=n/a planar=0.008 | body=n/a cross=n/a (n=100)
  turn_l_ceiling: net=0.165 ratio=0.984 wz_ratio=0.990 ripple=0.025 | drift=n/a planar=0.005 | body=n/a cross=n/a (n=100)
  turn_r_ceiling: net=-0.152 ratio=0.903 wz_ratio=0.908 ripple=0.043 | drift=n/a planar=0.007 | body=n/a cross=n/a (n=100)
    turn_l_w_max: net=0.186 ratio=0.886 wz_ratio=0.890 ripple=0.022 | drift=n/a planar=0.009 | body=n/a cross=n/a (n=100)
    turn_r_w_max: net=-0.181 ratio=0.862 wz_ratio=0.865 ripple=0.028 | drift=n/a planar=0.010 | body=n/a cross=n/a (n=100)

by category (10 waves, 1000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: n/a):
  turn_w_min: n=200 surv=1.000 (turn_l_w_min=1.00 turn_r_w_min=1.00) | ratio=n/a achieved=0.003 tripod=0.766 | closed=n/a progress=0.049 net=-0.004 | drift=0.038 dir_err=n/a
  turn_0p08: n=200 surv=1.000 (turn_l_0p08=1.00 turn_r_0p08=1.00) | ratio=n/a achieved=0.002 tripod=0.772 | closed=n/a progress=0.078 net=-0.006 | drift=0.039 dir_err=n/a
  turn_0p15: n=200 surv=1.000 (turn_l_0p15=1.00 turn_r_0p15=1.00) | ratio=n/a achieved=0.003 tripod=0.786 | closed=n/a progress=0.141 net=-0.013 | drift=0.042 dir_err=n/a
  turn_ceiling: n=200 surv=1.000 (turn_l_ceiling=1.00 turn_r_ceiling=1.00) | ratio=n/a achieved=0.004 tripod=0.783 | closed=n/a progress=0.156 net=-0.015 | drift=0.042 dir_err=n/a
  turn_w_max: n=200 surv=1.000 (turn_l_w_max=1.00 turn_r_w_max=1.00) | ratio=n/a achieved=0.001 tripod=0.796 | closed=n/a progress=0.176 net=-0.039 | drift=0.046 dir_err=n/a

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
  turn_w_min: net_yaw_rate=0.002 net_yaw_ratio=1.000 wz_ratio=1.004 yaw_ripple=0.031 heading_drift=n/a planar_drift=0.006 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  turn_0p08: net_yaw_rate=-0.002 net_yaw_ratio=0.995 wz_ratio=1.000 yaw_ripple=0.032 heading_drift=n/a planar_drift=0.005 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  turn_0p15: net_yaw_rate=0.008 net_yaw_ratio=0.962 wz_ratio=0.968 yaw_ripple=0.037 heading_drift=n/a planar_drift=0.006 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  turn_ceiling: net_yaw_rate=0.006 net_yaw_ratio=0.948 wz_ratio=0.955 yaw_ripple=0.032 heading_drift=n/a planar_drift=0.006 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  turn_w_max: net_yaw_rate=0.010 net_yaw_ratio=0.871 wz_ratio=0.873 yaw_ripple=0.025 heading_drift=n/a planar_drift=0.009 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
