```
=== crab-hex gait eval ===
scenario   : field_hard__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6647042198842165  p25=0.6373886157880589  p75=0.6812891888305661
tippy_tap_fraction median=0.42712282640825616
slip_ratio         median=0.27383819404609705
tracking_ratio     median=0.8942333404705292  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.91
terminations={'schedule_complete': 91, 'fall': 9}
facing_error_rms   median=0.5949407701693934  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.7864126358977156 (n=100)
       fwd: cmd 0.25 -> achieved 0.241 m/s | tripod median=0.5858019945205296 (n=100)
     lat_l: cmd 0.22 -> achieved 0.220 m/s | tripod median=0.617108104300194 (n=100)
      diag: cmd 0.25 -> achieved 0.235 m/s | tripod median=0.5825295203010715 (n=100)
      back: cmd 0.22 -> achieved 0.207 m/s | tripod median=0.6189052298392753 (n=95)
  turn_l90: cmd 0.00 -> achieved -0.008 m/s | tripod median=0.7549272543445136 (n=93)
  turn_r90: cmd 0.00 -> achieved 0.010 m/s | tripod median=0.7646130306013882 (n=93)
  move_turn: cmd 0.22 -> achieved 0.148 m/s | tripod median=0.585133190247049 (n=93)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.018 initial=0.009 final=0.021 max=0.043 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a
       fwd: err rms=0.063 initial=0.021 final=0.075 max=0.118 closed=n/a | settle=0.000 (n=93/100) | progress=-0.002 net=0.010 yaw_rate_ratio=0.394
     lat_l: err rms=0.053 initial=0.073 final=0.025 max=0.102 closed=0.803 | settle=0.000 (n=100/100) | progress=0.002 net=0.005 yaw_rate_ratio=0.499
      diag: err rms=0.045 initial=0.024 final=0.045 max=0.094 closed=n/a | settle=0.000 (n=90/100) | progress=-0.028 net=0.003 yaw_rate_ratio=0.399
      back: err rms=0.073 initial=0.043 final=0.104 max=0.135 closed=0.649 | settle=0.000 (n=84/95) | progress=-0.003 net=0.013 yaw_rate_ratio=0.452
  turn_l90: err rms=0.749 initial=1.471 final=0.020 max=1.486 closed=0.986 | settle=7.940 (n=93/94) | progress=0.169 net=0.122 yaw_rate_ratio=0.639
  turn_r90: err rms=0.779 initial=1.559 final=0.042 max=1.567 closed=0.973 | settle=8.300 (n=92/93) | progress=0.172 net=0.126 yaw_rate_ratio=0.646
  move_turn: err rms=1.219 initial=1.529 final=0.944 max=1.536 closed=0.383 | settle=n/a (n=0/93) | progress=0.072 net=0.073 yaw_rate_ratio=0.414

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
