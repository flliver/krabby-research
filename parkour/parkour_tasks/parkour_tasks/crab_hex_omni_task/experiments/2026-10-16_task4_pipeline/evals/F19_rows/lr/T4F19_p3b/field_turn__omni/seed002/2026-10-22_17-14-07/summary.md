```
=== crab-hex gait eval ===
scenario   : field_turn__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7247705270210658  p25=0.6983692992505148  p75=0.7422551388646146
tippy_tap_fraction median=0.4298642533936652
slip_ratio         median=0.28242602567688013
tracking_ratio     median=0.9632155647578063  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.99
terminations={'schedule_complete': 99, 'fall': 1}
facing_error_rms   median=0.595632348038082  (rad, whole episode; n=100)

by hold:
       fwd: cmd 0.25 -> achieved 0.241 m/s | tripod median=0.6001478132656958 (n=100)
     stand: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.7442355848516239 (n=99)
  turn_l90: cmd 0.00 -> achieved -0.006 m/s | tripod median=0.7707880156253624 (n=99)
  turn_r90: cmd 0.00 -> achieved 0.010 m/s | tripod median=0.7663726864607776 (n=99)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
       fwd: err rms=0.060 initial=0.050 final=0.062 max=0.132 closed=n/a | settle=0.000 (n=97/100) | progress=-0.002 net=0.011 yaw_rate_ratio=0.537
     stand: err rms=0.046 initial=0.060 final=0.029 max=0.087 closed=0.859 | settle=0.000 (n=99/99) | progress=-0.001 net=0.005 yaw_rate_ratio=0.420
  turn_l90: err rms=0.817 initial=1.590 final=0.017 max=1.593 closed=0.989 | settle=8.400 (n=99/99) | progress=0.172 net=0.131 yaw_rate_ratio=0.659
  turn_r90: err rms=0.781 initial=1.563 final=0.037 max=1.565 closed=0.976 | settle=8.240 (n=99/99) | progress=0.172 net=0.128 yaw_rate_ratio=0.643

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
