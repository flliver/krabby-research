```
=== crab-hex gait eval ===
scenario   : field_hard__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-10-21_20-59-46_seed19_X05_c6x_A15pB/model_14995.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5966387355136318  p25=0.5575565206791112  p75=0.6346391000964346
tippy_tap_fraction median=0.39924812030075185
slip_ratio         median=0.26699334051506196
tracking_ratio     median=0.8648395822691294  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.6
terminations={'schedule_complete': 60, 'fall': 40}
facing_error_rms   median=0.6446908981768236  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved -0.000 m/s | tripod median=0.7581279614980122 (n=100)
       fwd: cmd 0.25 -> achieved 0.240 m/s | tripod median=0.5355157951895153 (n=100)
     lat_l: cmd 0.22 -> achieved 0.219 m/s | tripod median=0.7270402275131682 (n=99)
      diag: cmd 0.25 -> achieved 0.224 m/s | tripod median=0.5095589630543074 (n=94)
      back: cmd 0.22 -> achieved 0.202 m/s | tripod median=0.44559948765704716 (n=80)
  turn_l90: cmd 0.00 -> achieved 0.012 m/s | tripod median=0.6675171864645814 (n=70)
  turn_r90: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.7095264588705334 (n=69)
  move_turn: cmd 0.22 -> achieved 0.120 m/s | tripod median=0.44269155594790033 (n=67)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.021 initial=0.009 final=0.021 max=0.043 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a
       fwd: err rms=0.056 initial=0.021 final=0.041 max=0.118 closed=n/a | settle=0.000 (n=99/100) | progress=-0.002 net=0.006 yaw_rate_ratio=0.695
     lat_l: err rms=0.040 initial=0.039 final=0.018 max=0.081 closed=n/a | settle=0.000 (n=99/99) | progress=0.004 net=0.004 yaw_rate_ratio=0.593
      diag: err rms=0.051 initial=0.017 final=0.055 max=0.107 closed=n/a | settle=0.000 (n=84/96) | progress=-0.003 net=0.002 yaw_rate_ratio=0.440
      back: err rms=0.064 initial=0.053 final=0.081 max=0.134 closed=0.340 | settle=0.000 (n=76/82) | progress=-0.001 net=0.016 yaw_rate_ratio=0.596
  turn_l90: err rms=1.258 initial=1.654 final=0.888 max=1.666 closed=0.469 | settle=n/a (n=0/71) | progress=0.065 net=0.065 yaw_rate_ratio=0.411
  turn_r90: err rms=0.285 initial=0.686 final=0.025 max=0.701 closed=0.962 | settle=5.020 (n=67/69) | progress=0.106 net=0.057 yaw_rate_ratio=0.453
  move_turn: err rms=1.217 initial=1.547 final=0.969 max=1.549 closed=0.382 | settle=n/a (n=0/69) | progress=0.075 net=0.076 yaw_rate_ratio=0.425

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
