```
=== crab-hex gait eval ===
scenario   : field_turn__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-21_23-06-12_seed19_X06_p2a_A15pB/model_19994.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.72423431712624  p25=0.6993503108717098  p75=0.7377849547165917
tippy_tap_fraction median=0.4094969777686712
slip_ratio         median=0.2525677616722861
tracking_ratio     median=1.0437331845044797  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.95
terminations={'schedule_complete': 95, 'fall': 5}
facing_error_rms   median=0.5746378407587823  (rad, whole episode; n=100)

by hold:
       fwd: cmd 0.25 -> achieved 0.261 m/s | tripod median=0.5681402949715594 (n=100)
     stand: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.7918297843876846 (n=99)
  turn_l90: cmd 0.00 -> achieved -0.002 m/s | tripod median=0.7745978430912333 (n=98)
  turn_r90: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.7708561745359551 (n=95)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
       fwd: err rms=0.035 initial=0.009 final=0.045 max=0.080 closed=n/a | settle=0.000 (n=99/100) | progress=-0.002 net=0.003 yaw_rate_ratio=0.811
     stand: err rms=0.039 initial=0.045 final=0.027 max=0.083 closed=0.412 | settle=0.000 (n=98/99) | progress=0.001 net=0.004 yaw_rate_ratio=0.422
  turn_l90: err rms=0.794 initial=1.551 final=0.082 max=1.553 closed=0.946 | settle=9.210 (n=96/98) | progress=0.154 net=0.123 yaw_rate_ratio=0.600
  turn_r90: err rms=0.754 initial=1.489 final=0.111 max=1.491 closed=0.926 | settle=10.420 (n=87/95) | progress=0.137 net=0.115 yaw_rate_ratio=0.585

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
