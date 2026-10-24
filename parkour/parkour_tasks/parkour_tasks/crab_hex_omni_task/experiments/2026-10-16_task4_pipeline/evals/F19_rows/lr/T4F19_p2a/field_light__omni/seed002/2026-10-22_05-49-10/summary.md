```
=== crab-hex gait eval ===
scenario   : field_light__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-21_23-06-12_seed19_X06_p2a_A15pB/model_19994.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.724445822254491  p25=0.7129478206627584  p75=0.7327723229017896
tippy_tap_fraction median=0.40544242873009995
slip_ratio         median=0.2372076902855227
tracking_ratio     median=0.961721044798399  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.99
terminations={'schedule_complete': 99, 'fall': 1}
facing_error_rms   median=0.5852472136725262  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.8117702487466913 (n=100)
       fwd: cmd 0.25 -> achieved 0.262 m/s | tripod median=0.5940828946824579 (n=100)
     lat_l: cmd 0.22 -> achieved 0.217 m/s | tripod median=0.7176174016311933 (n=100)
      diag: cmd 0.25 -> achieved 0.239 m/s | tripod median=0.7296357824060082 (n=100)
      back: cmd 0.22 -> achieved 0.230 m/s | tripod median=0.7074630859536009 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.003 m/s | tripod median=0.7911941900467607 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.000 m/s | tripod median=0.7916719452454515 (n=99)
  move_turn: cmd 0.22 -> achieved 0.173 m/s | tripod median=0.5937396921394741 (n=99)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.021 initial=0.007 final=0.027 max=0.047 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.053 initial=0.029 final=0.053 max=0.102 closed=n/a | settle=0.000 (n=100/100) | progress=-0.003 net=0.001 yaw_rate_ratio=0.284
     lat_l: err rms=0.040 initial=0.053 final=0.029 max=0.082 closed=n/a | settle=0.000 (n=100/100) | progress=-0.002 net=0.004 yaw_rate_ratio=0.445
      diag: err rms=0.043 initial=0.029 final=0.036 max=0.087 closed=n/a | settle=0.000 (n=100/100) | progress=-0.005 net=0.001 yaw_rate_ratio=0.290
      back: err rms=0.037 initial=0.037 final=0.029 max=0.082 closed=n/a | settle=0.000 (n=99/100) | progress=-0.006 net=0.005 yaw_rate_ratio=0.384
  turn_l90: err rms=0.845 initial=1.572 final=0.092 max=1.588 closed=0.942 | settle=9.780 (n=96/100) | progress=0.150 net=0.124 yaw_rate_ratio=0.588
  turn_r90: err rms=0.753 initial=1.479 final=0.110 max=1.487 closed=0.925 | settle=10.280 (n=95/99) | progress=0.138 net=0.114 yaw_rate_ratio=0.590
  move_turn: err rms=1.084 initial=1.461 final=0.741 max=1.464 closed=0.493 | settle=n/a (n=0/99) | progress=0.090 net=0.090 yaw_rate_ratio=0.393

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
