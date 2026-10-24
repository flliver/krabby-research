```
=== crab-hex gait eval ===
scenario   : field_light__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_03-27-53_seed19_X07_p3a_A15pB/model_22993.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6763452313204223  p25=0.659013419632914  p75=0.6938679421546159
tippy_tap_fraction median=0.37519716088328076
slip_ratio         median=0.2286930673584267
tracking_ratio     median=0.963117840740834  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.82
terminations={'schedule_complete': 82, 'fall': 18}
facing_error_rms   median=0.5993565018406424  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.7292702689067256 (n=100)
       fwd: cmd 0.25 -> achieved 0.254 m/s | tripod median=0.6626878570557522 (n=100)
     lat_l: cmd 0.22 -> achieved 0.212 m/s | tripod median=0.6640300258789911 (n=98)
      diag: cmd 0.25 -> achieved 0.249 m/s | tripod median=0.6077262564719513 (n=98)
      back: cmd 0.22 -> achieved 0.230 m/s | tripod median=0.6578523525216406 (n=93)
  turn_l90: cmd 0.00 -> achieved -0.008 m/s | tripod median=0.7770253061368767 (n=87)
  turn_r90: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.7692006009633463 (n=86)
  move_turn: cmd 0.22 -> achieved 0.179 m/s | tripod median=0.5874549009985205 (n=85)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.018 initial=0.008 final=0.019 max=0.046 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=0.000 yaw_rate_ratio=n/a
       fwd: err rms=0.060 initial=0.020 final=0.091 max=0.121 closed=n/a | settle=0.000 (n=92/100) | progress=-0.005 net=-0.001 yaw_rate_ratio=0.399
     lat_l: err rms=0.058 initial=0.092 final=0.021 max=0.129 closed=0.865 | settle=0.000 (n=98/98) | progress=0.004 net=0.010 yaw_rate_ratio=0.373
      diag: err rms=0.035 initial=0.021 final=0.032 max=0.078 closed=n/a | settle=0.000 (n=96/98) | progress=-0.004 net=0.003 yaw_rate_ratio=0.576
      back: err rms=0.061 initial=0.031 final=0.099 max=0.125 closed=0.615 | settle=0.000 (n=87/95) | progress=-0.006 net=0.008 yaw_rate_ratio=0.486
  turn_l90: err rms=0.976 initial=1.671 final=0.155 max=1.687 closed=0.907 | settle=11.550 (n=38/87) | progress=0.127 net=0.125 yaw_rate_ratio=0.560
  turn_r90: err rms=0.713 initial=1.416 final=0.129 max=1.426 closed=0.906 | settle=11.160 (n=61/86) | progress=0.115 net=0.107 yaw_rate_ratio=0.548
  move_turn: err rms=1.058 initial=1.442 final=0.702 max=1.453 closed=0.520 | settle=n/a (n=0/85) | progress=0.094 net=0.095 yaw_rate_ratio=0.432

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
