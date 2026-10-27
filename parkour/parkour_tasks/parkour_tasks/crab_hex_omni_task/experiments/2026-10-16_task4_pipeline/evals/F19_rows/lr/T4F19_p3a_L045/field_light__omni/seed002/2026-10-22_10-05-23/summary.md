```
=== crab-hex gait eval ===
scenario   : field_light__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_03-27-53_seed19_X07_p3a_A15pB/model_22993.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7564585872415421  p25=0.7390549639377845  p75=0.7650922879421905
tippy_tap_fraction median=0.3783084835876098
slip_ratio         median=0.2291498352043737
tracking_ratio     median=0.9439319377034243  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}
facing_error_rms   median=0.6004692479838744  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.8193267515425264 (n=100)
       fwd: cmd 0.25 -> achieved 0.253 m/s | tripod median=0.7519476626459851 (n=100)
     lat_l: cmd 0.22 -> achieved 0.208 m/s | tripod median=0.7107522290152937 (n=100)
      diag: cmd 0.25 -> achieved 0.239 m/s | tripod median=0.6750927006617663 (n=100)
      back: cmd 0.22 -> achieved 0.228 m/s | tripod median=0.7345447429733478 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.009 m/s | tripod median=0.8143760871780825 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.8012837878347103 (n=100)
  move_turn: cmd 0.22 -> achieved 0.172 m/s | tripod median=0.6946803883511901 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.019 initial=0.009 final=0.018 max=0.044 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.058 initial=0.018 final=0.095 max=0.120 closed=n/a | settle=0.000 (n=93/100) | progress=-0.005 net=-0.003 yaw_rate_ratio=0.331
     lat_l: err rms=0.059 initial=0.095 final=0.022 max=0.130 closed=0.868 | settle=0.000 (n=100/100) | progress=0.005 net=0.010 yaw_rate_ratio=0.415
      diag: err rms=0.030 initial=0.022 final=0.027 max=0.069 closed=n/a | settle=0.000 (n=99/100) | progress=-0.014 net=0.002 yaw_rate_ratio=0.571
      back: err rms=0.064 initial=0.027 final=0.103 max=0.138 closed=0.772 | settle=0.000 (n=93/100) | progress=-0.004 net=0.014 yaw_rate_ratio=0.409
  turn_l90: err rms=0.973 initial=1.675 final=0.130 max=1.696 closed=0.921 | settle=11.580 (n=73/100) | progress=0.135 net=0.128 yaw_rate_ratio=0.584
  turn_r90: err rms=0.720 initial=1.441 final=0.115 max=1.453 closed=0.920 | settle=11.270 (n=88/100) | progress=0.122 net=0.110 yaw_rate_ratio=0.571
  move_turn: err rms=1.055 initial=1.457 final=0.639 max=1.466 closed=0.558 | settle=n/a (n=0/100) | progress=0.101 net=0.102 yaw_rate_ratio=0.419

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
