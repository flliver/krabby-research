```
=== crab-hex gait eval ===
scenario   : field_light__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_03-27-53_seed19_X07_p3a_A15pB/model_22993.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.729179288385774  p25=0.7168073411246546  p75=0.7390125500095645
tippy_tap_fraction median=0.36750669334049946
slip_ratio         median=0.2279741972712437
tracking_ratio     median=0.9547227529682047  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.97
terminations={'schedule_complete': 97, 'fall': 3}
facing_error_rms   median=0.6029634576410972  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.000 m/s | tripod median=0.7879113608379769 (n=100)
       fwd: cmd 0.25 -> achieved 0.254 m/s | tripod median=0.7243104834544131 (n=100)
     lat_l: cmd 0.22 -> achieved 0.209 m/s | tripod median=0.691039234853629 (n=100)
      diag: cmd 0.25 -> achieved 0.243 m/s | tripod median=0.6466443292749728 (n=99)
      back: cmd 0.22 -> achieved 0.231 m/s | tripod median=0.7030422546008732 (n=99)
  turn_l90: cmd 0.00 -> achieved -0.008 m/s | tripod median=0.8019542148896911 (n=97)
  turn_r90: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.7904164276824955 (n=97)
  move_turn: cmd 0.22 -> achieved 0.176 m/s | tripod median=0.6497543222783705 (n=97)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.018 initial=0.009 final=0.015 max=0.046 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.056 initial=0.015 final=0.101 max=0.118 closed=n/a | settle=0.000 (n=89/100) | progress=-0.004 net=-0.001 yaw_rate_ratio=0.345
     lat_l: err rms=0.064 initial=0.100 final=0.023 max=0.132 closed=0.738 | settle=0.000 (n=100/100) | progress=0.004 net=0.010 yaw_rate_ratio=0.333
      diag: err rms=0.032 initial=0.023 final=0.028 max=0.074 closed=n/a | settle=0.000 (n=99/100) | progress=-0.028 net=0.003 yaw_rate_ratio=0.503
      back: err rms=0.067 initial=0.028 final=0.110 max=0.135 closed=0.860 | settle=0.000 (n=86/99) | progress=-0.005 net=0.012 yaw_rate_ratio=0.412
  turn_l90: err rms=0.983 initial=1.678 final=0.151 max=1.698 closed=0.911 | settle=11.580 (n=48/97) | progress=0.130 net=0.128 yaw_rate_ratio=0.574
  turn_r90: err rms=0.725 initial=1.421 final=0.122 max=1.440 closed=0.916 | settle=11.420 (n=79/97) | progress=0.119 net=0.109 yaw_rate_ratio=0.553
  move_turn: err rms=1.042 initial=1.447 final=0.643 max=1.462 closed=0.550 | settle=n/a (n=0/97) | progress=0.099 net=0.099 yaw_rate_ratio=0.429

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
