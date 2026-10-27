```
=== crab-hex gait eval ===
scenario   : field_light__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_27000.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.676775301583384  p25=0.6612478922420367  p75=0.6889228669600185
tippy_tap_fraction median=0.3166436896854242
slip_ratio         median=0.23506156571088785
tracking_ratio     median=0.9533214189801891  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.98
terminations={'schedule_complete': 98, 'fall': 2}
facing_error_rms   median=0.5990150329466454  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7666346506923666 (n=100)
       fwd: cmd 0.25 -> achieved 0.248 m/s | tripod median=0.576947281448176 (n=100)
     lat_l: cmd 0.22 -> achieved 0.227 m/s | tripod median=0.6160138449353855 (n=100)
      diag: cmd 0.25 -> achieved 0.252 m/s | tripod median=0.6191172622090636 (n=100)
      back: cmd 0.22 -> achieved 0.219 m/s | tripod median=0.6678011448549471 (n=99)
  turn_l90: cmd 0.00 -> achieved -0.009 m/s | tripod median=0.7608306415185072 (n=99)
  turn_r90: cmd 0.00 -> achieved 0.011 m/s | tripod median=0.7754801991507987 (n=99)
  move_turn: cmd 0.22 -> achieved 0.168 m/s | tripod median=0.5400795670022596 (n=99)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.018 initial=0.007 final=0.017 max=0.040 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.058 initial=0.018 final=0.042 max=0.120 closed=n/a | settle=0.000 (n=99/100) | progress=-0.001 net=0.006 yaw_rate_ratio=0.471
     lat_l: err rms=0.045 initial=0.041 final=0.032 max=0.086 closed=0.866 | settle=0.000 (n=100/100) | progress=-0.000 net=0.003 yaw_rate_ratio=0.421
      diag: err rms=0.037 initial=0.032 final=0.033 max=0.079 closed=n/a | settle=0.000 (n=100/100) | progress=-0.007 net=0.003 yaw_rate_ratio=0.658
      back: err rms=0.051 initial=0.032 final=0.068 max=0.108 closed=n/a | settle=0.000 (n=98/99) | progress=-0.000 net=0.011 yaw_rate_ratio=0.682
  turn_l90: err rms=0.762 initial=1.503 final=0.019 max=1.506 closed=0.987 | settle=8.000 (n=99/99) | progress=0.168 net=0.124 yaw_rate_ratio=0.629
  turn_r90: err rms=0.773 initial=1.560 final=0.030 max=1.565 closed=0.981 | settle=7.920 (n=99/99) | progress=0.177 net=0.129 yaw_rate_ratio=0.649
  move_turn: err rms=1.241 initial=1.561 final=0.967 max=1.566 closed=0.376 | settle=n/a (n=0/99) | progress=0.073 net=0.074 yaw_rate_ratio=0.424

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
