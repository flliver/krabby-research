```
=== crab-hex gait eval ===
scenario   : field_light__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-10-21_20-59-46_seed19_X05_c6x_A15pB/model_14995.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6615065510696441  p25=0.6493648086443982  p75=0.6754210283866676
tippy_tap_fraction median=0.3838218038471173
slip_ratio         median=0.24997741226432102
tracking_ratio     median=0.8915657263761259  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.98
terminations={'schedule_complete': 98, 'fall': 2}
facing_error_rms   median=0.652569094820789  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.7652466681160084 (n=100)
       fwd: cmd 0.25 -> achieved 0.243 m/s | tripod median=0.5489201302218187 (n=100)
     lat_l: cmd 0.22 -> achieved 0.226 m/s | tripod median=0.7881085520266526 (n=100)
      diag: cmd 0.25 -> achieved 0.241 m/s | tripod median=0.6564006158550367 (n=99)
      back: cmd 0.22 -> achieved 0.208 m/s | tripod median=0.5327298825788686 (n=99)
  turn_l90: cmd 0.00 -> achieved 0.012 m/s | tripod median=0.7271653863094012 (n=99)
  turn_r90: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.7527189110956656 (n=99)
  move_turn: cmd 0.22 -> achieved 0.124 m/s | tripod median=0.4673841678100392 (n=99)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.020 initial=0.009 final=0.016 max=0.040 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a
       fwd: err rms=0.053 initial=0.016 final=0.047 max=0.106 closed=n/a | settle=0.000 (n=99/100) | progress=0.004 net=0.006 yaw_rate_ratio=0.673
     lat_l: err rms=0.037 initial=0.048 final=0.017 max=0.087 closed=0.123 | settle=0.000 (n=99/100) | progress=0.016 net=0.006 yaw_rate_ratio=0.384
      diag: err rms=0.042 initial=0.016 final=0.043 max=0.086 closed=n/a | settle=0.000 (n=99/99) | progress=-0.013 net=0.001 yaw_rate_ratio=0.491
      back: err rms=0.052 initial=0.044 final=0.083 max=0.117 closed=n/a | settle=0.000 (n=95/99) | progress=-0.008 net=0.016 yaw_rate_ratio=0.447
  turn_l90: err rms=1.242 initial=1.653 final=0.844 max=1.663 closed=0.488 | settle=n/a (n=0/99) | progress=0.068 net=0.068 yaw_rate_ratio=0.413
  turn_r90: err rms=0.299 initial=0.727 final=0.023 max=0.735 closed=0.967 | settle=5.140 (n=99/99) | progress=0.110 net=0.060 yaw_rate_ratio=0.452
  move_turn: err rms=1.195 initial=1.555 final=0.935 max=1.557 closed=0.398 | settle=n/a (n=0/99) | progress=0.077 net=0.078 yaw_rate_ratio=0.424

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
