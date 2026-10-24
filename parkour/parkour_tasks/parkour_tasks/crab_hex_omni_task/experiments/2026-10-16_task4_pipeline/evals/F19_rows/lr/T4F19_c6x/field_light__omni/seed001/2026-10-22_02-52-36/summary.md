```
=== crab-hex gait eval ===
scenario   : field_light__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-10-21_20-59-46_seed19_X05_c6x_A15pB/model_14995.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6637423647734182  p25=0.6452382170874097  p75=0.6752222000865529
tippy_tap_fraction median=0.39086294416243655
slip_ratio         median=0.24627310022652224
tracking_ratio     median=0.8950446272513601  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.95
terminations={'schedule_complete': 95, 'fall': 5}
facing_error_rms   median=0.6481051587997908  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.000 m/s | tripod median=0.764487238528329 (n=100)
       fwd: cmd 0.25 -> achieved 0.244 m/s | tripod median=0.545971122353499 (n=100)
     lat_l: cmd 0.22 -> achieved 0.227 m/s | tripod median=0.7877675498506657 (n=100)
      diag: cmd 0.25 -> achieved 0.239 m/s | tripod median=0.6634129689560575 (n=100)
      back: cmd 0.22 -> achieved 0.208 m/s | tripod median=0.5368335663835528 (n=98)
  turn_l90: cmd 0.00 -> achieved 0.012 m/s | tripod median=0.7323202179769855 (n=95)
  turn_r90: cmd 0.00 -> achieved -0.000 m/s | tripod median=0.7529343849024083 (n=95)
  move_turn: cmd 0.22 -> achieved 0.123 m/s | tripod median=0.48312567254563016 (n=95)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.019 initial=0.011 final=0.017 max=0.043 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.051 initial=0.017 final=0.042 max=0.101 closed=n/a | settle=0.000 (n=100/100) | progress=0.007 net=0.006 yaw_rate_ratio=0.696
     lat_l: err rms=0.034 initial=0.043 final=0.018 max=0.079 closed=n/a | settle=0.000 (n=100/100) | progress=0.003 net=0.004 yaw_rate_ratio=0.698
      diag: err rms=0.045 initial=0.018 final=0.046 max=0.088 closed=n/a | settle=0.000 (n=99/100) | progress=-0.008 net=0.002 yaw_rate_ratio=0.476
      back: err rms=0.058 initial=0.046 final=0.083 max=0.118 closed=n/a | settle=0.000 (n=96/98) | progress=-0.002 net=0.016 yaw_rate_ratio=0.450
  turn_l90: err rms=1.231 initial=1.655 final=0.847 max=1.671 closed=0.492 | settle=n/a (n=0/95) | progress=0.069 net=0.068 yaw_rate_ratio=0.417
  turn_r90: err rms=0.316 initial=0.724 final=0.021 max=0.742 closed=0.971 | settle=5.240 (n=95/95) | progress=0.109 net=0.059 yaw_rate_ratio=0.444
  move_turn: err rms=1.184 initial=1.554 final=0.915 max=1.558 closed=0.415 | settle=n/a (n=0/95) | progress=0.081 net=0.082 yaw_rate_ratio=0.422

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
