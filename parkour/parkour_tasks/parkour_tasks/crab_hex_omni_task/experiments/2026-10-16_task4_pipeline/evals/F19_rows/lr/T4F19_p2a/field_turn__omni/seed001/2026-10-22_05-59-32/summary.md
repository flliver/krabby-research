```
=== crab-hex gait eval ===
scenario   : field_turn__omni  (Isaac-Crab-Hex-Omni-Walk-Play-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-21_23-06-12_seed19_X06_p2a_A15pB/model_19994.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.7314522274617165  p25=0.7031187234170286  p75=0.7540798045935784
tippy_tap_fraction median=0.41867573322343565
slip_ratio         median=0.24569818755894057
tracking_ratio     median=1.039605478697224  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.97
terminations={'schedule_complete': 97, 'fall': 3}
facing_error_rms   median=0.5705497444822958  (rad, whole episode; n=100)

by hold:
       fwd: cmd 0.25 -> achieved 0.260 m/s | tripod median=0.577907609669096 (n=100)
     stand: cmd 0.00 -> achieved 0.005 m/s | tripod median=0.7903981292233608 (n=99)
  turn_l90: cmd 0.00 -> achieved -0.003 m/s | tripod median=0.7855940527490554 (n=99)
  turn_r90: cmd 0.00 -> achieved 0.000 m/s | tripod median=0.7847355572589434 (n=98)

facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of the initial error closed, n/a when there was none; ratio is wobble-sensitive):
       fwd: err rms=0.040 initial=0.007 final=0.051 max=0.080 closed=n/a | settle=0.000 (n=96/100) | progress=-0.014 net=0.004 yaw_rate_ratio=0.871
     stand: err rms=0.049 initial=0.052 final=0.034 max=0.098 closed=0.666 | settle=0.000 (n=100/100) | progress=-0.002 net=0.004 yaw_rate_ratio=0.427
  turn_l90: err rms=0.797 initial=1.549 final=0.093 max=1.553 closed=0.941 | settle=9.340 (n=95/99) | progress=0.154 net=0.121 yaw_rate_ratio=0.606
  turn_r90: err rms=0.738 initial=1.477 final=0.109 max=1.478 closed=0.926 | settle=10.030 (n=92/98) | progress=0.139 net=0.114 yaw_rate_ratio=0.587

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
