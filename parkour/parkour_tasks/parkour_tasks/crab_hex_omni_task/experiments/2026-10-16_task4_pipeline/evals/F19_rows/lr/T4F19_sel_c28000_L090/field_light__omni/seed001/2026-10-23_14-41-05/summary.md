```
=== crab-hex gait eval ===
scenario   : field_light__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_28000.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6837654552948641  p25=0.6716342589338202  p75=0.6926853799772711
tippy_tap_fraction median=0.32441471571906355
slip_ratio         median=0.2263897131286586
tracking_ratio     median=0.9570519591119604  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.95
terminations={'schedule_complete': 95, 'fall': 5}
facing_error_rms   median=0.5980284073454498  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.7602143653359835 (n=100)
       fwd: cmd 0.25 -> achieved 0.248 m/s | tripod median=0.5750159998130979 (n=100)
     lat_l: cmd 0.22 -> achieved 0.229 m/s | tripod median=0.6566144968267316 (n=100)
      diag: cmd 0.25 -> achieved 0.253 m/s | tripod median=0.6463239819968987 (n=99)
      back: cmd 0.22 -> achieved 0.221 m/s | tripod median=0.6676827354256507 (n=98)
  turn_l90: cmd 0.00 -> achieved -0.012 m/s | tripod median=0.7731514621586404 (n=98)
  turn_r90: cmd 0.00 -> achieved 0.011 m/s | tripod median=0.7858333869153539 (n=98)
  move_turn: cmd 0.22 -> achieved 0.166 m/s | tripod median=0.5509003399617332 (n=98)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.022 initial=0.010 final=0.029 max=0.050 closed=n/a | settle=0.000 (n=100/100) | progress=-0.033 net=-0.001 yaw_rate_ratio=0.792
       fwd: err rms=0.063 initial=0.028 final=0.043 max=0.127 closed=n/a | settle=0.000 (n=97/100) | progress=-0.001 net=0.005 yaw_rate_ratio=0.479
     lat_l: err rms=0.035 initial=0.043 final=0.023 max=0.078 closed=0.905 | settle=0.000 (n=100/100) | progress=-0.001 net=0.004 yaw_rate_ratio=0.616
      diag: err rms=0.038 initial=0.024 final=0.041 max=0.080 closed=n/a | settle=0.000 (n=98/100) | progress=-0.045 net=0.001 yaw_rate_ratio=0.589
      back: err rms=0.053 initial=0.037 final=0.078 max=0.119 closed=0.555 | settle=0.000 (n=94/99) | progress=-0.003 net=0.014 yaw_rate_ratio=0.601
  turn_l90: err rms=0.770 initial=1.496 final=0.021 max=1.511 closed=0.986 | settle=8.080 (n=98/98) | progress=0.167 net=0.124 yaw_rate_ratio=0.632
  turn_r90: err rms=0.786 initial=1.565 final=0.038 max=1.573 closed=0.976 | settle=8.170 (n=98/98) | progress=0.172 net=0.128 yaw_rate_ratio=0.643
  move_turn: err rms=1.217 initial=1.538 final=0.968 max=1.544 closed=0.373 | settle=n/a (n=0/98) | progress=0.072 net=0.073 yaw_rate_ratio=0.431

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
