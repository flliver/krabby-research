```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-05_12-03-33/model_4999.pt
episodes   : 100  unscored: 2
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.39536624741324944  p25=0.3185590369372472  p75=0.4619660734814243
tippy_tap_fraction median=0.3587324625060474
slip_ratio         median=0.28555804307833854
tracking_ratio     median=0.6861225936156253  (achieved/commanded vx, walking holds; n=74)
schedule_completion_rate=0.13
terminations={'fall': 87, 'schedule_complete': 13}

by hold:
     stand: cmd 0.00 -> achieved 0.078 m/s | tripod median=0.3705938500604977 (n=98)
     creep: cmd 0.25 -> achieved 0.181 m/s | tripod median=0.4458238892233237 (n=71)
       low: cmd 0.35 -> achieved 0.196 m/s | tripod median=0.5198536163692798 (n=29)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
