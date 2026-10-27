```
=== crab-hex gait eval ===
scenario   : field_light__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_28000.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7070285677413957  p25=0.6980428447209454  p75=0.7160715466113396
tippy_tap_fraction median=0.38461163827041245
slip_ratio         median=0.25544875807722167
tracking_ratio     median=0.9344415216455269  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}
facing_error_rms   median=0.5859906611815932  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.7774406519129787 (n=100)
       fwd: cmd 0.25 -> achieved 0.243 m/s | tripod median=0.5850167717875501 (n=100)
     lat_l: cmd 0.22 -> achieved 0.229 m/s | tripod median=0.6606596687255586 (n=100)
      diag: cmd 0.25 -> achieved 0.249 m/s | tripod median=0.6759702765109892 (n=100)
      back: cmd 0.22 -> achieved 0.215 m/s | tripod median=0.6861755427135872 (n=100)
  turn_l90: cmd 0.00 -> achieved -0.010 m/s | tripod median=0.784081133663021 (n=100)
  turn_r90: cmd 0.00 -> achieved 0.010 m/s | tripod median=0.8024052074001721 (n=100)
  move_turn: cmd 0.22 -> achieved 0.155 m/s | tripod median=0.6029889743223835 (n=100)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.021 initial=0.009 final=0.025 max=0.047 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.000 yaw_rate_ratio=n/a
       fwd: err rms=0.057 initial=0.025 final=0.043 max=0.113 closed=n/a | settle=0.000 (n=99/100) | progress=0.001 net=0.008 yaw_rate_ratio=0.332
     lat_l: err rms=0.038 initial=0.044 final=0.027 max=0.080 closed=0.739 | settle=0.000 (n=100/100) | progress=0.001 net=0.003 yaw_rate_ratio=0.466
      diag: err rms=0.043 initial=0.026 final=0.049 max=0.082 closed=n/a | settle=0.000 (n=100/100) | progress=-0.159 net=-0.001 yaw_rate_ratio=0.793
      back: err rms=0.057 initial=0.049 final=0.085 max=0.114 closed=n/a | settle=0.000 (n=96/100) | progress=-0.007 net=0.015 yaw_rate_ratio=0.473
  turn_l90: err rms=0.751 initial=1.486 final=0.024 max=1.490 closed=0.984 | settle=7.970 (n=100/100) | progress=0.168 net=0.123 yaw_rate_ratio=0.646
  turn_r90: err rms=0.772 initial=1.559 final=0.038 max=1.563 closed=0.976 | settle=7.890 (n=100/100) | progress=0.178 net=0.127 yaw_rate_ratio=0.655
  move_turn: err rms=1.190 initial=1.536 final=0.892 max=1.543 closed=0.424 | settle=n/a (n=0/100) | progress=0.080 net=0.082 yaw_rate_ratio=0.411

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
