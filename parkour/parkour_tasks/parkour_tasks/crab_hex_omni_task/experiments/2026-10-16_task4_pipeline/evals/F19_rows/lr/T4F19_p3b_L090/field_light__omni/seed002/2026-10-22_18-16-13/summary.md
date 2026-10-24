```
=== crab-hex gait eval ===
scenario   : field_light__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6805572173004711  p25=0.6675676514662746  p75=0.6947709048073982
tippy_tap_fraction median=0.3668618323175121
slip_ratio         median=0.23666513016373114
tracking_ratio     median=0.9459082764553118  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.92
terminations={'schedule_complete': 92, 'fall': 8}
facing_error_rms   median=0.603037822023714  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.766161073226822 (n=100)
       fwd: cmd 0.25 -> achieved 0.248 m/s | tripod median=0.5768992413620467 (n=100)
     lat_l: cmd 0.22 -> achieved 0.227 m/s | tripod median=0.6657669334521473 (n=100)
      diag: cmd 0.25 -> achieved 0.250 m/s | tripod median=0.627336890925575 (n=99)
      back: cmd 0.22 -> achieved 0.217 m/s | tripod median=0.6617420176347527 (n=99)
  turn_l90: cmd 0.00 -> achieved -0.010 m/s | tripod median=0.7801256912808086 (n=99)
  turn_r90: cmd 0.00 -> achieved 0.012 m/s | tripod median=0.7772633127030835 (n=99)
  move_turn: cmd 0.22 -> achieved 0.168 m/s | tripod median=0.553570025843316 (n=99)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.018 initial=0.009 final=0.022 max=0.043 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.056 initial=0.023 final=0.043 max=0.112 closed=n/a | settle=0.000 (n=98/100) | progress=-0.001 net=0.003 yaw_rate_ratio=0.473
     lat_l: err rms=0.037 initial=0.040 final=0.027 max=0.077 closed=0.690 | settle=0.000 (n=100/100) | progress=0.003 net=0.005 yaw_rate_ratio=0.618
      diag: err rms=0.045 initial=0.027 final=0.051 max=0.089 closed=n/a | settle=0.000 (n=97/99) | progress=-0.000 net=-0.001 yaw_rate_ratio=0.399
      back: err rms=0.057 initial=0.051 final=0.079 max=0.124 closed=0.571 | settle=0.000 (n=97/99) | progress=0.001 net=0.015 yaw_rate_ratio=0.682
  turn_l90: err rms=0.761 initial=1.490 final=0.021 max=1.500 closed=0.986 | settle=7.960 (n=99/99) | progress=0.169 net=0.124 yaw_rate_ratio=0.641
  turn_r90: err rms=0.786 initial=1.566 final=0.046 max=1.571 closed=0.971 | settle=8.300 (n=99/99) | progress=0.172 net=0.127 yaw_rate_ratio=0.647
  move_turn: err rms=1.237 initial=1.532 final=0.999 max=1.537 closed=0.355 | settle=n/a (n=0/99) | progress=0.067 net=0.068 yaw_rate_ratio=0.422

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
