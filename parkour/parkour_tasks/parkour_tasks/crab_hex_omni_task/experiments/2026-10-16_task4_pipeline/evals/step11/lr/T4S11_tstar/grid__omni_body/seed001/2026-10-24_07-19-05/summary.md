```
=== crab-hex gait eval ===
scenario   : grid__omni_body  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt
episodes   : 2000  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.710531012358693  p25=0.6541869052764571  p75=0.7544228587154197
tippy_tap_fraction median=0.4578313253012048
slip_ratio         median=0.24444374213758352
tracking_ratio     median=0.891169682936593  (achieved/commanded vx, walking holds; n=1600)
schedule_completion_rate=1.0
terminations={'schedule_complete': 2000}
facing_error_rms   median=0.21363543390965678  (rad, whole episode; n=2000)

by hold:
    settle: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.7722520012334645 (n=1961)
   g_v0_w0: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.790186608330925 (n=100)
  g_v0_w0p08: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.7748894210444022 (n=100)
  g_v0_w0p15: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.7776590027888532 (n=100)
  g_v0_w0p3: cmd 0.00 -> achieved -0.010 m/s | tripod median=0.785904080356775 (n=100)
  g_v0p22_fwd_w0: cmd 0.22 -> achieved 0.228 m/s | tripod median=0.6956935696283242 (n=100)
  g_v0p22_fwd_w0p08: cmd 0.22 -> achieved 0.196 m/s | tripod median=0.694103855559586 (n=100)
  g_v0p22_fwd_w0p15: cmd 0.22 -> achieved 0.172 m/s | tripod median=0.6659296961857761 (n=100)
  g_v0p22_fwd_w0p3: cmd 0.22 -> achieved 0.161 m/s | tripod median=0.6535506886922393 (n=100)
  g_v0p22_lat_w0: cmd 0.22 -> achieved 0.235 m/s | tripod median=0.6789804317472077 (n=100)
  g_v0p22_lat_w0p08: cmd 0.22 -> achieved 0.226 m/s | tripod median=0.6935019068498908 (n=100)
  g_v0p22_lat_w0p15: cmd 0.22 -> achieved 0.219 m/s | tripod median=0.7189805319890767 (n=100)
  g_v0p22_lat_w0p3: cmd 0.22 -> achieved 0.201 m/s | tripod median=0.7578237132017898 (n=100)
  g_v0p3_fwd_w0: cmd 0.30 -> achieved 0.277 m/s | tripod median=0.5656620821092505 (n=100)
  g_v0p3_fwd_w0p08: cmd 0.30 -> achieved 0.239 m/s | tripod median=0.46238517876326074 (n=100)
  g_v0p3_fwd_w0p15: cmd 0.30 -> achieved 0.221 m/s | tripod median=0.43004512805354844 (n=100)
  g_v0p3_fwd_w0p3: cmd 0.30 -> achieved 0.209 m/s | tripod median=0.4003075192374881 (n=100)
  g_v0p3_lat_w0: cmd 0.30 -> achieved 0.282 m/s | tripod median=0.720148711004726 (n=100)
  g_v0p3_lat_w0p08: cmd 0.30 -> achieved 0.270 m/s | tripod median=0.7197116669545651 (n=100)
  g_v0p3_lat_w0p15: cmd 0.30 -> achieved 0.261 m/s | tripod median=0.7549942661471012 (n=100)
  g_v0p3_lat_w0p3: cmd 0.30 -> achieved 0.239 m/s | tripod median=0.7466936342282039 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
    settle: err rms=0.014 initial=0.008 final=0.013 max=0.028 closed=n/a | settle=0.000 (n=2000/2000) | progress=n/a net=0.002 yaw_rate_ratio=n/a
   g_v0_w0: err rms=0.024 initial=0.013 final=0.027 max=0.055 closed=n/a | settle=0.000 (n=100/100) | progress=-0.001 net=0.000 yaw_rate_ratio=0.570
  g_v0_w0p08: err rms=0.128 initial=0.012 final=0.144 max=0.206 closed=n/a | settle=11.580 (n=60/100) | progress=0.078 net=-0.006 yaw_rate_ratio=0.672
  g_v0_w0p15: err rms=0.177 initial=0.012 final=0.195 max=0.270 closed=n/a | settle=11.810 (n=14/100) | progress=0.145 net=0.011 yaw_rate_ratio=0.649
  g_v0_w0p3: err rms=0.530 initial=0.012 final=0.600 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.181 net=-0.046 yaw_rate_ratio=0.664
  g_v0p22_fwd_w0: err rms=0.044 initial=0.015 final=0.029 max=0.100 closed=n/a | settle=0.000 (n=100/100) | progress=0.000 net=0.001 yaw_rate_ratio=0.423
  g_v0p22_fwd_w0p08: err rms=0.243 initial=0.015 final=0.381 max=0.392 closed=n/a | settle=n/a (n=0/100) | progress=0.054 net=-0.023 yaw_rate_ratio=0.293
  g_v0p22_fwd_w0p15: err rms=0.480 initial=0.014 final=0.600 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.063 net=-0.047 yaw_rate_ratio=0.314
  g_v0p22_fwd_w0p3: err rms=0.556 initial=0.011 final=0.600 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.065 net=-0.048 yaw_rate_ratio=0.336
  g_v0p22_lat_w0: err rms=0.018 initial=0.011 final=0.015 max=0.041 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
  g_v0p22_lat_w0p08: err rms=0.152 initial=0.012 final=0.137 max=0.235 closed=n/a | settle=11.570 (n=64/100) | progress=0.080 net=0.011 yaw_rate_ratio=0.550
  g_v0p22_lat_w0p15: err rms=0.242 initial=0.015 final=0.275 max=0.342 closed=n/a | settle=n/a (n=0/100) | progress=0.137 net=-0.018 yaw_rate_ratio=0.569
  g_v0p22_lat_w0p3: err rms=0.541 initial=0.013 final=0.600 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.167 net=-0.048 yaw_rate_ratio=0.617
  g_v0p3_fwd_w0: err rms=0.037 initial=0.011 final=0.050 max=0.083 closed=n/a | settle=0.000 (n=100/100) | progress=-0.000 net=0.000 yaw_rate_ratio=0.385
  g_v0p3_fwd_w0p08: err rms=0.382 initial=0.012 final=0.578 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.034 net=0.045 yaw_rate_ratio=0.354
  g_v0p3_fwd_w0p15: err rms=0.509 initial=0.015 final=0.600 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.041 net=-0.047 yaw_rate_ratio=0.341
  g_v0p3_fwd_w0p3: err rms=0.560 initial=0.016 final=0.600 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.042 net=-0.047 yaw_rate_ratio=0.351
  g_v0p3_lat_w0: err rms=0.017 initial=0.012 final=0.012 max=0.044 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
  g_v0p3_lat_w0p08: err rms=0.148 initial=0.012 final=0.144 max=0.222 closed=n/a | settle=11.660 (n=58/100) | progress=0.079 net=-0.008 yaw_rate_ratio=0.554
  g_v0p3_lat_w0p15: err rms=0.289 initial=0.011 final=0.354 max=0.405 closed=n/a | settle=n/a (n=0/100) | progress=0.130 net=-0.025 yaw_rate_ratio=0.537
  g_v0p3_lat_w0p3: err rms=0.544 initial=0.012 final=0.600 max=0.600 closed=n/a | settle=n/a (n=0/100) | progress=0.151 net=-0.048 yaw_rate_ratio=0.567

body yaw by hold (window [4 s, hold_s); medians; net = net yaw rate rad/s, ratio = net / wz_cmd SIGNED (the gated turn number), wz_ratio = mean wz / wz_cmd; ripple = std of the 1-s mean wz; drift = |mean wz| on wz = 0 holds; planar = xy drift m/s on pure turns; body = along-bearing speed / |v_cmd|, cross = mean |perp| m/s):
          settle: net=n/a ratio=n/a wz_ratio=n/a ripple=n/a | drift=n/a planar=n/a | body=n/a cross=n/a (n=0)
         g_v0_w0: net=0.001 ratio=n/a wz_ratio=n/a ripple=0.016 | drift=0.002 planar=n/a | body=n/a cross=n/a (n=100)
      g_v0_w0p08: net=0.079 ratio=0.992 wz_ratio=0.998 ripple=0.029 | drift=n/a planar=0.004 | body=n/a cross=n/a (n=100)
      g_v0_w0p15: net=0.146 ratio=0.976 wz_ratio=0.981 ripple=0.027 | drift=n/a planar=0.005 | body=n/a cross=n/a (n=100)
       g_v0_w0p3: net=0.185 ratio=0.616 wz_ratio=0.619 ripple=0.028 | drift=n/a planar=0.015 | body=n/a cross=n/a (n=100)
  g_v0p22_fwd_w0: net=-0.007 ratio=n/a wz_ratio=n/a ripple=0.041 | drift=0.008 planar=n/a | body=1.031 cross=0.024 (n=100)
  g_v0p22_fwd_w0p08: net=0.053 ratio=0.657 wz_ratio=0.652 ripple=0.027 | drift=n/a planar=n/a | body=0.854 cross=0.025 (n=100)
  g_v0p22_fwd_w0p15: net=0.060 ratio=0.401 wz_ratio=0.397 ripple=0.037 | drift=n/a planar=n/a | body=0.729 cross=0.026 (n=100)
  g_v0p22_fwd_w0p3: net=0.058 ratio=0.192 wz_ratio=0.189 ripple=0.026 | drift=n/a planar=n/a | body=0.696 cross=0.026 (n=100)
  g_v0p22_lat_w0: net=-0.001 ratio=n/a wz_ratio=n/a ripple=0.014 | drift=0.002 planar=n/a | body=1.074 cross=0.025 (n=100)
  g_v0p22_lat_w0p08: net=0.086 ratio=1.076 wz_ratio=1.083 ripple=0.030 | drift=n/a planar=n/a | body=1.018 cross=0.038 (n=100)
  g_v0p22_lat_w0p15: net=0.151 ratio=1.006 wz_ratio=1.012 ripple=0.029 | drift=n/a planar=n/a | body=0.991 cross=0.038 (n=100)
  g_v0p22_lat_w0p3: net=0.176 ratio=0.586 wz_ratio=0.588 ripple=0.029 | drift=n/a planar=n/a | body=0.902 cross=0.052 (n=100)
   g_v0p3_fwd_w0: net=-0.002 ratio=n/a wz_ratio=n/a ripple=0.021 | drift=0.003 planar=n/a | body=0.917 cross=0.026 (n=100)
  g_v0p3_fwd_w0p08: net=0.037 ratio=0.457 wz_ratio=0.448 ripple=0.035 | drift=n/a planar=n/a | body=0.763 cross=0.029 (n=100)
  g_v0p3_fwd_w0p15: net=0.043 ratio=0.286 wz_ratio=0.279 ripple=0.032 | drift=n/a planar=n/a | body=0.696 cross=0.031 (n=100)
  g_v0p3_fwd_w0p3: net=0.039 ratio=0.130 wz_ratio=0.126 ripple=0.033 | drift=n/a planar=n/a | body=0.672 cross=0.034 (n=100)
   g_v0p3_lat_w0: net=0.001 ratio=n/a wz_ratio=n/a ripple=0.014 | drift=0.002 planar=n/a | body=0.940 cross=0.028 (n=100)
  g_v0p3_lat_w0p08: net=0.084 ratio=1.056 wz_ratio=1.060 ripple=0.023 | drift=n/a planar=n/a | body=0.890 cross=0.036 (n=100)
  g_v0p3_lat_w0p15: net=0.143 ratio=0.956 wz_ratio=0.957 ripple=0.028 | drift=n/a planar=n/a | body=0.856 cross=0.037 (n=100)
  g_v0p3_lat_w0p3: net=0.160 ratio=0.533 wz_ratio=0.533 ripple=0.032 | drift=n/a planar=n/a | body=0.787 cross=0.054 (n=100)

by category (20 waves, 2000 episodes, one own hold each; surv = own-hold survival; medians pooled over both hands; trained: n/a):
    g_v0_w0: n=100 surv=1.000 (g_v0_w0=1.00) | ratio=n/a achieved=0.003 tripod=0.790 | closed=n/a progress=-0.001 net=0.000 | drift=0.034 dir_err=n/a
  g_v0_w0p08: n=100 surv=1.000 (g_v0_w0p08=1.00) | ratio=n/a achieved=0.001 tripod=0.775 | closed=n/a progress=0.078 net=-0.006 | drift=0.038 dir_err=n/a
  g_v0_w0p15: n=100 surv=1.000 (g_v0_w0p15=1.00) | ratio=n/a achieved=0.002 tripod=0.778 | closed=n/a progress=0.145 net=0.011 | drift=0.041 dir_err=n/a
  g_v0_w0p3: n=100 surv=1.000 (g_v0_w0p3=1.00) | ratio=n/a achieved=-0.010 tripod=0.786 | closed=n/a progress=0.181 net=-0.046 | drift=0.053 dir_err=n/a
  g_v0p22_fwd_w0: n=100 surv=1.000 (g_v0p22_fwd_w0=1.00) | ratio=1.034 achieved=0.228 tripod=0.696 | closed=n/a progress=0.000 net=0.001 | drift=0.228 dir_err=0.010
  g_v0p22_fwd_w0p08: n=100 surv=1.000 (g_v0p22_fwd_w0p08=1.00) | ratio=0.891 achieved=0.196 tripod=0.694 | closed=n/a progress=0.054 net=-0.023 | drift=0.199 dir_err=0.019
  g_v0p22_fwd_w0p15: n=100 surv=1.000 (g_v0p22_fwd_w0p15=1.00) | ratio=0.783 achieved=0.172 tripod=0.666 | closed=n/a progress=0.063 net=-0.047 | drift=0.179 dir_err=0.016
  g_v0p22_fwd_w0p3: n=100 surv=1.000 (g_v0p22_fwd_w0p3=1.00) | ratio=0.730 achieved=0.161 tripod=0.654 | closed=n/a progress=0.065 net=-0.048 | drift=0.168 dir_err=0.014
  g_v0p22_lat_w0: n=100 surv=1.000 (g_v0p22_lat_w0=1.00) | ratio=1.069 achieved=0.235 tripod=0.679 | closed=n/a progress=n/a net=0.001 | drift=0.237 dir_err=0.008
  g_v0p22_lat_w0p08: n=100 surv=1.000 (g_v0p22_lat_w0p08=1.00) | ratio=1.029 achieved=0.226 tripod=0.694 | closed=n/a progress=0.080 net=0.011 | drift=0.230 dir_err=0.043
  g_v0p22_lat_w0p15: n=100 surv=1.000 (g_v0p22_lat_w0p15=1.00) | ratio=0.997 achieved=0.219 tripod=0.719 | closed=n/a progress=0.137 net=-0.018 | drift=0.224 dir_err=0.038
  g_v0p22_lat_w0p3: n=100 surv=1.000 (g_v0p22_lat_w0p3=1.00) | ratio=0.912 achieved=0.201 tripod=0.758 | closed=n/a progress=0.167 net=-0.048 | drift=0.213 dir_err=0.026
  g_v0p3_fwd_w0: n=100 surv=1.000 (g_v0p3_fwd_w0=1.00) | ratio=0.924 achieved=0.277 tripod=0.566 | closed=n/a progress=-0.000 net=0.000 | drift=0.278 dir_err=0.012
  g_v0p3_fwd_w0p08: n=100 surv=1.000 (g_v0p3_fwd_w0p08=1.00) | ratio=0.796 achieved=0.239 tripod=0.462 | closed=n/a progress=0.034 net=0.045 | drift=0.244 dir_err=0.011
  g_v0p3_fwd_w0p15: n=100 surv=1.000 (g_v0p3_fwd_w0p15=1.00) | ratio=0.738 achieved=0.221 tripod=0.430 | closed=n/a progress=0.041 net=-0.047 | drift=0.229 dir_err=0.017
  g_v0p3_fwd_w0p3: n=100 surv=1.000 (g_v0p3_fwd_w0p3=1.00) | ratio=0.696 achieved=0.209 tripod=0.400 | closed=n/a progress=0.042 net=-0.047 | drift=0.218 dir_err=0.020
  g_v0p3_lat_w0: n=100 surv=1.000 (g_v0p3_lat_w0=1.00) | ratio=0.940 achieved=0.282 tripod=0.720 | closed=n/a progress=n/a net=0.001 | drift=0.282 dir_err=0.011
  g_v0p3_lat_w0p08: n=100 surv=1.000 (g_v0p3_lat_w0p08=1.00) | ratio=0.901 achieved=0.270 tripod=0.720 | closed=n/a progress=0.079 net=-0.008 | drift=0.273 dir_err=0.045
  g_v0p3_lat_w0p15: n=100 surv=1.000 (g_v0p3_lat_w0p15=1.00) | ratio=0.869 achieved=0.261 tripod=0.755 | closed=n/a progress=0.130 net=-0.025 | drift=0.264 dir_err=0.035
  g_v0p3_lat_w0p3: n=100 surv=1.000 (g_v0p3_lat_w0p3=1.00) | ratio=0.798 achieved=0.239 tripod=0.747 | closed=n/a progress=0.151 net=-0.048 | drift=0.252 dir_err=0.028

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
    g_v0_w0: net_yaw_rate=0.001 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.016 heading_drift=0.002 planar_drift=n/a body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  g_v0_w0p08: net_yaw_rate=0.079 net_yaw_ratio=0.992 wz_ratio=0.998 yaw_ripple=0.029 heading_drift=n/a planar_drift=0.004 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  g_v0_w0p15: net_yaw_rate=0.146 net_yaw_ratio=0.976 wz_ratio=0.981 yaw_ripple=0.027 heading_drift=n/a planar_drift=0.005 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  g_v0_w0p3: net_yaw_rate=0.185 net_yaw_ratio=0.616 wz_ratio=0.619 yaw_ripple=0.028 heading_drift=n/a planar_drift=0.015 body_ratio=n/a cross_body=n/a adapter_clipped_frac=n/a
  g_v0p22_fwd_w0: net_yaw_rate=-0.007 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.041 heading_drift=0.008 planar_drift=n/a body_ratio=1.031 cross_body=0.024 adapter_clipped_frac=n/a
  g_v0p22_fwd_w0p08: net_yaw_rate=0.053 net_yaw_ratio=0.657 wz_ratio=0.652 yaw_ripple=0.027 heading_drift=n/a planar_drift=n/a body_ratio=0.854 cross_body=0.025 adapter_clipped_frac=n/a
  g_v0p22_fwd_w0p15: net_yaw_rate=0.060 net_yaw_ratio=0.401 wz_ratio=0.397 yaw_ripple=0.037 heading_drift=n/a planar_drift=n/a body_ratio=0.729 cross_body=0.026 adapter_clipped_frac=n/a
  g_v0p22_fwd_w0p3: net_yaw_rate=0.058 net_yaw_ratio=0.192 wz_ratio=0.189 yaw_ripple=0.026 heading_drift=n/a planar_drift=n/a body_ratio=0.696 cross_body=0.026 adapter_clipped_frac=n/a
  g_v0p22_lat_w0: net_yaw_rate=-0.001 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.014 heading_drift=0.002 planar_drift=n/a body_ratio=1.074 cross_body=0.025 adapter_clipped_frac=n/a
  g_v0p22_lat_w0p08: net_yaw_rate=0.086 net_yaw_ratio=1.076 wz_ratio=1.083 yaw_ripple=0.030 heading_drift=n/a planar_drift=n/a body_ratio=1.018 cross_body=0.038 adapter_clipped_frac=n/a
  g_v0p22_lat_w0p15: net_yaw_rate=0.151 net_yaw_ratio=1.006 wz_ratio=1.012 yaw_ripple=0.029 heading_drift=n/a planar_drift=n/a body_ratio=0.991 cross_body=0.038 adapter_clipped_frac=n/a
  g_v0p22_lat_w0p3: net_yaw_rate=0.176 net_yaw_ratio=0.586 wz_ratio=0.588 yaw_ripple=0.029 heading_drift=n/a planar_drift=n/a body_ratio=0.902 cross_body=0.052 adapter_clipped_frac=n/a
  g_v0p3_fwd_w0: net_yaw_rate=-0.002 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.021 heading_drift=0.003 planar_drift=n/a body_ratio=0.917 cross_body=0.026 adapter_clipped_frac=n/a
  g_v0p3_fwd_w0p08: net_yaw_rate=0.037 net_yaw_ratio=0.457 wz_ratio=0.448 yaw_ripple=0.035 heading_drift=n/a planar_drift=n/a body_ratio=0.763 cross_body=0.029 adapter_clipped_frac=n/a
  g_v0p3_fwd_w0p15: net_yaw_rate=0.043 net_yaw_ratio=0.286 wz_ratio=0.279 yaw_ripple=0.032 heading_drift=n/a planar_drift=n/a body_ratio=0.696 cross_body=0.031 adapter_clipped_frac=n/a
  g_v0p3_fwd_w0p3: net_yaw_rate=0.039 net_yaw_ratio=0.130 wz_ratio=0.126 yaw_ripple=0.033 heading_drift=n/a planar_drift=n/a body_ratio=0.672 cross_body=0.034 adapter_clipped_frac=n/a
  g_v0p3_lat_w0: net_yaw_rate=0.001 net_yaw_ratio=n/a wz_ratio=n/a yaw_ripple=0.014 heading_drift=0.002 planar_drift=n/a body_ratio=0.940 cross_body=0.028 adapter_clipped_frac=n/a
  g_v0p3_lat_w0p08: net_yaw_rate=0.084 net_yaw_ratio=1.056 wz_ratio=1.060 yaw_ripple=0.023 heading_drift=n/a planar_drift=n/a body_ratio=0.890 cross_body=0.036 adapter_clipped_frac=n/a
  g_v0p3_lat_w0p15: net_yaw_rate=0.143 net_yaw_ratio=0.956 wz_ratio=0.957 yaw_ripple=0.028 heading_drift=n/a planar_drift=n/a body_ratio=0.856 cross_body=0.037 adapter_clipped_frac=n/a
  g_v0p3_lat_w0p3: net_yaw_rate=0.160 net_yaw_ratio=0.533 wz_ratio=0.533 yaw_ripple=0.032 heading_drift=n/a planar_drift=n/a body_ratio=0.787 cross_body=0.054 adapter_clipped_frac=n/a

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
