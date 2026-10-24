```
=== crab-hex gait eval ===
scenario   : field_hard__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_03-27-53_seed19_X07_p3a_A15pB/model_22993.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.720141902882064  p25=0.6815049324172369  p75=0.7472769153486177
tippy_tap_fraction median=0.41259654574523186
slip_ratio         median=0.2410661068132533
tracking_ratio     median=0.9086085259444336  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.83
terminations={'schedule_complete': 83, 'fall': 17}
facing_error_rms   median=0.5977688592458028  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.000 m/s | tripod median=0.8351796095111177 (n=100)
       fwd: cmd 0.25 -> achieved 0.248 m/s | tripod median=0.7650972185967395 (n=100)
     lat_l: cmd 0.22 -> achieved 0.201 m/s | tripod median=0.6988845895197591 (n=100)
      diag: cmd 0.25 -> achieved 0.226 m/s | tripod median=0.607419805950825 (n=99)
      back: cmd 0.22 -> achieved 0.223 m/s | tripod median=0.6832746424253592 (n=93)
  turn_l90: cmd 0.00 -> achieved -0.006 m/s | tripod median=0.7704187263291432 (n=89)
  turn_r90: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.7731559940168072 (n=89)
  move_turn: cmd 0.22 -> achieved 0.159 m/s | tripod median=0.6849312171427333 (n=88)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.021 initial=0.012 final=0.019 max=0.051 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.055 initial=0.019 final=0.098 max=0.119 closed=n/a | settle=0.000 (n=92/100) | progress=-0.011 net=0.000 yaw_rate_ratio=0.299
     lat_l: err rms=0.064 initial=0.098 final=0.025 max=0.124 closed=0.837 | settle=0.000 (n=100/100) | progress=0.010 net=0.010 yaw_rate_ratio=0.364
      diag: err rms=0.039 initial=0.025 final=0.037 max=0.087 closed=n/a | settle=0.000 (n=91/100) | progress=-0.024 net=0.001 yaw_rate_ratio=0.477
      back: err rms=0.066 initial=0.034 final=0.088 max=0.136 closed=0.327 | settle=0.000 (n=87/96) | progress=-0.002 net=0.012 yaw_rate_ratio=0.364
  turn_l90: err rms=0.967 initial=1.657 final=0.127 max=1.673 closed=0.925 | settle=11.500 (n=65/89) | progress=0.131 net=0.128 yaw_rate_ratio=0.574
  turn_r90: err rms=0.717 initial=1.443 final=0.119 max=1.451 closed=0.917 | settle=10.600 (n=79/89) | progress=0.127 net=0.110 yaw_rate_ratio=0.576
  move_turn: err rms=1.060 initial=1.452 final=0.653 max=1.465 closed=0.549 | settle=n/a (n=0/88) | progress=0.100 net=0.100 yaw_rate_ratio=0.423

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
