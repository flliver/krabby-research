```
=== crab-hex gait eval ===
scenario   : ring_obst__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-10-21_20-59-46_seed19_X05_c6x_A15pB/model_14995.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6554926905322406  p25=0.6332012463241359  p75=0.6679510625350511
tippy_tap_fraction median=0.39290212306585104
slip_ratio         median=0.24408824610369556
tracking_ratio     median=0.8946841227432147  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.93
terminations={'schedule_complete': 93, 'fall': 7}
facing_error_rms   median=0.654558120965741  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.7605288696185832 (n=100)
       fwd: cmd 0.25 -> achieved 0.241 m/s | tripod median=0.5544248788568218 (n=100)
     lat_l: cmd 0.22 -> achieved 0.223 m/s | tripod median=0.7803902210264211 (n=100)
      diag: cmd 0.25 -> achieved 0.242 m/s | tripod median=0.6471134716134961 (n=100)
      back: cmd 0.22 -> achieved 0.211 m/s | tripod median=0.528309054495149 (n=98)
  turn_l90: cmd 0.00 -> achieved 0.012 m/s | tripod median=0.7021917537147268 (n=98)
  turn_r90: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.7428670385091388 (n=98)
  move_turn: cmd 0.22 -> achieved 0.125 m/s | tripod median=0.45907534074923406 (n=98)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.020 initial=0.011 final=0.016 max=0.042 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.051 initial=0.016 final=0.044 max=0.104 closed=n/a | settle=0.000 (n=100/100) | progress=0.007 net=0.006 yaw_rate_ratio=0.762
     lat_l: err rms=0.034 initial=0.044 final=0.017 max=0.077 closed=n/a | settle=0.000 (n=100/100) | progress=-0.003 net=0.005 yaw_rate_ratio=0.431
      diag: err rms=0.043 initial=0.016 final=0.038 max=0.085 closed=n/a | settle=0.000 (n=100/100) | progress=0.040 net=0.001 yaw_rate_ratio=0.395
      back: err rms=0.054 initial=0.036 final=0.090 max=0.114 closed=n/a | settle=0.000 (n=95/98) | progress=-0.007 net=0.016 yaw_rate_ratio=0.558
  turn_l90: err rms=1.236 initial=1.660 final=0.865 max=1.676 closed=0.471 | settle=n/a (n=0/98) | progress=0.066 net=0.065 yaw_rate_ratio=0.419
  turn_r90: err rms=0.295 initial=0.707 final=0.022 max=0.718 closed=0.968 | settle=5.160 (n=98/98) | progress=0.109 net=0.058 yaw_rate_ratio=0.455
  move_turn: err rms=1.201 initial=1.555 final=0.942 max=1.562 closed=0.388 | settle=n/a (n=0/98) | progress=0.075 net=0.076 yaw_rate_ratio=0.420

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
