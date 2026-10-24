```
=== crab-hex gait eval ===
scenario   : field_hard__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6500759768700848  p25=0.6148429507915144  p75=0.6764050608522675
tippy_tap_fraction median=0.42875939849624056
slip_ratio         median=0.27820691639640327
tracking_ratio     median=0.8781556269117439  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.86
terminations={'schedule_complete': 86, 'fall': 14}
facing_error_rms   median=0.5962092653488065  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.788885844075504 (n=100)
       fwd: cmd 0.25 -> achieved 0.239 m/s | tripod median=0.5719007272180774 (n=100)
     lat_l: cmd 0.22 -> achieved 0.214 m/s | tripod median=0.598305853294659 (n=100)
      diag: cmd 0.25 -> achieved 0.233 m/s | tripod median=0.5741826416563913 (n=97)
      back: cmd 0.22 -> achieved 0.206 m/s | tripod median=0.6343212984414756 (n=95)
  turn_l90: cmd 0.00 -> achieved -0.008 m/s | tripod median=0.7267447909476312 (n=93)
  turn_r90: cmd 0.00 -> achieved 0.010 m/s | tripod median=0.7370400748952842 (n=93)
  move_turn: cmd 0.22 -> achieved 0.155 m/s | tripod median=0.5825238323240879 (n=91)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.020 initial=0.007 final=0.022 max=0.046 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.000 yaw_rate_ratio=n/a
       fwd: err rms=0.060 initial=0.022 final=0.080 max=0.122 closed=n/a | settle=0.000 (n=91/100) | progress=-0.003 net=0.008 yaw_rate_ratio=0.421
     lat_l: err rms=0.050 initial=0.079 final=0.029 max=0.106 closed=0.748 | settle=0.000 (n=100/100) | progress=0.004 net=0.007 yaw_rate_ratio=0.528
      diag: err rms=0.055 initial=0.028 final=0.056 max=0.105 closed=n/a | settle=0.000 (n=94/98) | progress=-0.001 net=0.003 yaw_rate_ratio=0.449
      back: err rms=0.068 initial=0.054 final=0.084 max=0.133 closed=0.567 | settle=0.000 (n=90/95) | progress=-0.001 net=0.009 yaw_rate_ratio=0.512
  turn_l90: err rms=0.761 initial=1.492 final=0.026 max=1.497 closed=0.983 | settle=8.220 (n=92/93) | progress=0.165 net=0.123 yaw_rate_ratio=0.639
  turn_r90: err rms=0.793 initial=1.559 final=0.051 max=1.564 closed=0.968 | settle=8.500 (n=91/93) | progress=0.166 net=0.125 yaw_rate_ratio=0.641
  move_turn: err rms=1.198 initial=1.525 final=0.923 max=1.533 closed=0.399 | settle=n/a (n=0/92) | progress=0.076 net=0.077 yaw_rate_ratio=0.418

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
