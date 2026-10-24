```
=== crab-hex gait eval ===
scenario   : field_light__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_28000.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6589991033880325  p25=0.6476707216444313  p75=0.674050564449218
tippy_tap_fraction median=0.31748685545613875
slip_ratio         median=0.23188800382700406
tracking_ratio     median=0.9631479109028522  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.89
terminations={'schedule_complete': 89, 'fall': 11}
facing_error_rms   median=0.6001007113733947  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.751143729459733 (n=100)
       fwd: cmd 0.25 -> achieved 0.250 m/s | tripod median=0.5778953589171927 (n=100)
     lat_l: cmd 0.22 -> achieved 0.228 m/s | tripod median=0.6520531149004237 (n=98)
      diag: cmd 0.25 -> achieved 0.254 m/s | tripod median=0.6368564353557844 (n=98)
      back: cmd 0.22 -> achieved 0.222 m/s | tripod median=0.6240620855507686 (n=96)
  turn_l90: cmd 0.00 -> achieved -0.014 m/s | tripod median=0.7557576337448049 (n=96)
  turn_r90: cmd 0.00 -> achieved 0.010 m/s | tripod median=0.7556358258917577 (n=96)
  move_turn: cmd 0.22 -> achieved 0.176 m/s | tripod median=0.49336825395082196 (n=96)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.018 initial=0.009 final=0.020 max=0.043 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.062 initial=0.020 final=0.052 max=0.126 closed=n/a | settle=0.000 (n=96/100) | progress=-0.002 net=0.001 yaw_rate_ratio=0.472
     lat_l: err rms=0.036 initial=0.051 final=0.019 max=0.085 closed=0.906 | settle=0.000 (n=99/99) | progress=0.029 net=0.006 yaw_rate_ratio=0.484
      diag: err rms=0.039 initial=0.019 final=0.041 max=0.083 closed=n/a | settle=0.000 (n=97/98) | progress=-0.017 net=0.002 yaw_rate_ratio=0.376
      back: err rms=0.052 initial=0.040 final=0.063 max=0.112 closed=n/a | settle=0.000 (n=96/96) | progress=-0.002 net=0.012 yaw_rate_ratio=0.630
  turn_l90: err rms=0.783 initial=1.509 final=0.021 max=1.520 closed=0.986 | settle=8.250 (n=96/96) | progress=0.164 net=0.125 yaw_rate_ratio=0.622
  turn_r90: err rms=0.784 initial=1.562 final=0.047 max=1.564 closed=0.970 | settle=8.220 (n=96/96) | progress=0.172 net=0.127 yaw_rate_ratio=0.640
  move_turn: err rms=1.225 initial=1.528 final=1.020 max=1.536 closed=0.340 | settle=n/a (n=0/96) | progress=0.065 net=0.066 yaw_rate_ratio=0.447

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
