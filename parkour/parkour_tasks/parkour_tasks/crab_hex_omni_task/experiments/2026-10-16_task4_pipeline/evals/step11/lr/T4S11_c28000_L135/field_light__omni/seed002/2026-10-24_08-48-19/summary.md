```
=== crab-hex gait eval ===
scenario   : field_light__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_28000.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6601872982200092  p25=0.6496690251457882  p75=0.6704216537762613
tippy_tap_fraction median=0.32269611834962
slip_ratio         median=0.2253469597271242
tracking_ratio     median=0.9591637570742957  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.86
terminations={'schedule_complete': 86, 'fall': 14}
facing_error_rms   median=0.5996366615652381  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7453334430665461 (n=100)
       fwd: cmd 0.25 -> achieved 0.251 m/s | tripod median=0.5754294832452693 (n=100)
     lat_l: cmd 0.22 -> achieved 0.226 m/s | tripod median=0.648569692228709 (n=100)
      diag: cmd 0.25 -> achieved 0.254 m/s | tripod median=0.6377887796593648 (n=99)
      back: cmd 0.22 -> achieved 0.222 m/s | tripod median=0.638003463408044 (n=98)
  turn_l90: cmd 0.00 -> achieved -0.012 m/s | tripod median=0.7478817777302749 (n=98)
  turn_r90: cmd 0.00 -> achieved 0.009 m/s | tripod median=0.7606139085971428 (n=97)
  move_turn: cmd 0.22 -> achieved 0.172 m/s | tripod median=0.4542145340904776 (n=97)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.020 initial=0.008 final=0.026 max=0.046 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.002 yaw_rate_ratio=n/a
       fwd: err rms=0.055 initial=0.025 final=0.037 max=0.124 closed=n/a | settle=0.000 (n=99/100) | progress=-0.001 net=0.001 yaw_rate_ratio=0.529
     lat_l: err rms=0.037 initial=0.037 final=0.022 max=0.080 closed=0.778 | settle=0.000 (n=100/100) | progress=0.000 net=0.005 yaw_rate_ratio=0.411
      diag: err rms=0.035 initial=0.022 final=0.030 max=0.078 closed=n/a | settle=0.000 (n=98/99) | progress=-0.010 net=0.001 yaw_rate_ratio=0.641
      back: err rms=0.050 initial=0.030 final=0.062 max=0.107 closed=0.691 | settle=0.000 (n=94/98) | progress=-0.005 net=0.008 yaw_rate_ratio=0.559
  turn_l90: err rms=0.788 initial=1.508 final=0.016 max=1.516 closed=0.989 | settle=8.220 (n=96/98) | progress=0.166 net=0.126 yaw_rate_ratio=0.625
  turn_r90: err rms=0.791 initial=1.569 final=0.046 max=1.574 closed=0.971 | settle=8.270 (n=96/97) | progress=0.171 net=0.127 yaw_rate_ratio=0.635
  move_turn: err rms=1.218 initial=1.532 final=1.000 max=1.534 closed=0.349 | settle=n/a (n=0/97) | progress=0.065 net=0.067 yaw_rate_ratio=0.437

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
