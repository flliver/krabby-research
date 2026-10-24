```
=== crab-hex gait eval ===
scenario   : field_light__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_28000.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6830243546208918  p25=0.6698169697294315  p75=0.6921458431360218
tippy_tap_fraction median=0.32451638373470193
slip_ratio         median=0.23239355847252963
tracking_ratio     median=0.9624229328200705  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.95
terminations={'schedule_complete': 95, 'fall': 5}
facing_error_rms   median=0.5995571001109847  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7525954533248405 (n=100)
       fwd: cmd 0.25 -> achieved 0.250 m/s | tripod median=0.5928161337922837 (n=100)
     lat_l: cmd 0.22 -> achieved 0.229 m/s | tripod median=0.6653497909113198 (n=100)
      diag: cmd 0.25 -> achieved 0.253 m/s | tripod median=0.6546169265985635 (n=100)
      back: cmd 0.22 -> achieved 0.220 m/s | tripod median=0.6555876374030835 (n=99)
  turn_l90: cmd 0.00 -> achieved -0.012 m/s | tripod median=0.7761085628608213 (n=98)
  turn_r90: cmd 0.00 -> achieved 0.010 m/s | tripod median=0.7769618722269062 (n=98)
  move_turn: cmd 0.22 -> achieved 0.168 m/s | tripod median=0.5284859978437824 (n=98)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.022 initial=0.007 final=0.024 max=0.048 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.000 yaw_rate_ratio=n/a
       fwd: err rms=0.058 initial=0.025 final=0.044 max=0.116 closed=n/a | settle=0.000 (n=98/100) | progress=0.001 net=0.005 yaw_rate_ratio=0.509
     lat_l: err rms=0.032 initial=0.045 final=0.024 max=0.081 closed=0.435 | settle=0.000 (n=100/100) | progress=-0.006 net=0.005 yaw_rate_ratio=0.561
      diag: err rms=0.038 initial=0.024 final=0.045 max=0.083 closed=n/a | settle=0.000 (n=99/100) | progress=-0.004 net=-0.001 yaw_rate_ratio=0.428
      back: err rms=0.053 initial=0.044 final=0.071 max=0.119 closed=0.407 | settle=0.000 (n=97/99) | progress=-0.003 net=0.014 yaw_rate_ratio=0.546
  turn_l90: err rms=0.780 initial=1.498 final=0.018 max=1.513 closed=0.988 | settle=8.130 (n=98/99) | progress=0.169 net=0.125 yaw_rate_ratio=0.635
  turn_r90: err rms=0.786 initial=1.565 final=0.037 max=1.567 closed=0.976 | settle=8.120 (n=98/98) | progress=0.174 net=0.128 yaw_rate_ratio=0.644
  move_turn: err rms=1.224 initial=1.541 final=0.971 max=1.552 closed=0.375 | settle=n/a (n=0/98) | progress=0.072 net=0.073 yaw_rate_ratio=0.433

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
