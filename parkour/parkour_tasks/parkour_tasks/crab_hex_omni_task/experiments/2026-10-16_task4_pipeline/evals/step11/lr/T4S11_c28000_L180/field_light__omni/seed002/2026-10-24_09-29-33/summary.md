```
=== crab-hex gait eval ===
scenario   : field_light__omni  (Isaac-Crab-Hex-Omni-Student-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_28000.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6136050208966167  p25=0.5621964631935196  p75=0.6319364565878189
tippy_tap_fraction median=0.3229569892473118
slip_ratio         median=0.2341921874192361
tracking_ratio     median=0.9537487937709971  (achieved/commanded vx, walking holds; n=99)
schedule_completion_rate=0.52
terminations={'schedule_complete': 52, 'fall': 48}
facing_error_rms   median=0.5948499230130966  (rad, whole episode; n=100)

by hold:
     stand: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.7303878112273472 (n=100)
       fwd: cmd 0.25 -> achieved 0.251 m/s | tripod median=0.5497171232812103 (n=99)
     lat_l: cmd 0.22 -> achieved 0.219 m/s | tripod median=0.6291516751261543 (n=96)
      diag: cmd 0.25 -> achieved 0.250 m/s | tripod median=0.5997641978457247 (n=93)
      back: cmd 0.22 -> achieved 0.221 m/s | tripod median=0.5046783905850135 (n=83)
  turn_l90: cmd 0.00 -> achieved -0.014 m/s | tripod median=0.728204536997386 (n=70)
  turn_r90: cmd 0.00 -> achieved 0.009 m/s | tripod median=0.7308887819209374 (n=69)
  move_turn: cmd 0.22 -> achieved 0.187 m/s | tripod median=0.41860756196784454 (n=69)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
     stand: err rms=0.018 initial=0.010 final=0.023 max=0.043 closed=n/a | settle=0.000 (n=100/100) | progress=n/a net=-0.001 yaw_rate_ratio=n/a
       fwd: err rms=0.054 initial=0.023 final=0.045 max=0.121 closed=n/a | settle=0.000 (n=98/99) | progress=0.001 net=0.004 yaw_rate_ratio=0.862
     lat_l: err rms=0.039 initial=0.047 final=0.022 max=0.087 closed=0.861 | settle=0.000 (n=98/98) | progress=0.001 net=0.005 yaw_rate_ratio=0.568
      diag: err rms=0.038 initial=0.022 final=0.034 max=0.086 closed=n/a | settle=0.000 (n=91/95) | progress=-0.007 net=0.002 yaw_rate_ratio=0.342
      back: err rms=0.057 initial=0.033 final=0.073 max=0.117 closed=0.238 | settle=0.000 (n=73/85) | progress=-0.003 net=0.010 yaw_rate_ratio=0.627
  turn_l90: err rms=0.797 initial=1.505 final=0.023 max=1.523 closed=0.985 | settle=8.330 (n=68/71) | progress=0.164 net=0.125 yaw_rate_ratio=0.605
  turn_r90: err rms=0.799 initial=1.560 final=0.036 max=1.562 closed=0.976 | settle=8.440 (n=68/69) | progress=0.168 net=0.127 yaw_rate_ratio=0.630
  move_turn: err rms=1.255 initial=1.533 final=1.041 max=1.541 closed=0.311 | settle=n/a (n=0/69) | progress=0.063 net=0.064 yaw_rate_ratio=0.457

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
