```
=== crab-hex gait eval ===
scenario   : field_hard__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_29000.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6442766231647956  p25=0.6063285744987851  p75=0.6717852330903784
tippy_tap_fraction median=0.39054235703223167
slip_ratio         median=0.27590300960451364
tracking_ratio     median=0.8728518985874703  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.9
terminations={'schedule_complete': 90, 'fall': 10}
facing_error_rms   median=0.5953315566718929  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.7842430122954216 (n=100)
       fwd: cmd 0.25 -> achieved 0.242 m/s | tripod median=0.5877300265060592 (n=100)
     lat_l: cmd 0.22 -> achieved 0.214 m/s | tripod median=0.567231276602472 (n=100)
      diag: cmd 0.25 -> achieved 0.226 m/s | tripod median=0.5505587644976346 (n=97)
      back: cmd 0.22 -> achieved 0.207 m/s | tripod median=0.6228573139607644 (n=93)
  turn_l90: cmd 0.00 -> achieved -0.008 m/s | tripod median=0.7397008066529289 (n=90)
  turn_r90: cmd 0.00 -> achieved 0.008 m/s | tripod median=0.7544365045399053 (n=90)
  move_turn: cmd 0.22 -> achieved 0.152 m/s | tripod median=0.546146585164438 (n=90)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.020 initial=0.007 final=0.024 max=0.044 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.000 yaw_rate_ratio=n/a
       fwd: err rms=0.060 initial=0.023 final=0.052 max=0.123 closed=n/a | settle=0.000 (n=94/100) | progress=-0.003 net=0.005 yaw_rate_ratio=0.535
     lat_l: err rms=0.047 initial=0.051 final=0.040 max=0.095 closed=0.775 | settle=0.000 (n=99/100) | progress=0.000 net=0.004 yaw_rate_ratio=0.561
      diag: err rms=0.055 initial=0.040 final=0.053 max=0.107 closed=0.938 | settle=0.000 (n=91/99) | progress=-0.003 net=0.003 yaw_rate_ratio=0.460
      back: err rms=0.068 initial=0.052 final=0.082 max=0.134 closed=0.708 | settle=0.000 (n=89/95) | progress=0.001 net=0.014 yaw_rate_ratio=0.568
  turn_l90: err rms=0.768 initial=1.491 final=0.023 max=1.497 closed=0.985 | settle=8.120 (n=89/90) | progress=0.166 net=0.124 yaw_rate_ratio=0.638
  turn_r90: err rms=0.784 initial=1.562 final=0.043 max=1.566 closed=0.972 | settle=8.420 (n=87/90) | progress=0.171 net=0.126 yaw_rate_ratio=0.638
  move_turn: err rms=1.203 initial=1.528 final=0.924 max=1.535 closed=0.398 | settle=n/a (n=0/90) | progress=0.074 net=0.076 yaw_rate_ratio=0.415

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
