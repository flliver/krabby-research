```
=== crab-hex gait eval ===
scenario   : slow__side  (Isaac-Crab-Hex-Side-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_side_teacher/2026-09-11_14-50-01/model_9998.pt
episodes   : 100  unscored: 1
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.3843378733529011  p25=0.35173250360548913  p75=0.4308559113997427
tippy_tap_fraction median=0.4132457580733443
slip_ratio         median=0.3875111566840006
tracking_ratio     median=0.6501575896068224  (achieved/commanded vx, walking holds; n=98)
schedule_completion_rate=0.88
terminations={'schedule_complete': 88, 'fall': 12}

by hold:
     stand: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.0 (n=87)
     creep: cmd 0.25 -> achieved 0.192 m/s | tripod median=0.5478822633218119 (n=98)
       low: cmd 0.35 -> achieved 0.182 m/s | tripod median=0.56704344023772 (n=92)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
