```
=== crab-hex gait eval ===
scenario   : slow__A15pB  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-06_18-32-32/model_9998.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6599272414361662  p25=0.6258896456573072  p75=0.6797407203705288
tippy_tap_fraction median=0.20274995274995275
slip_ratio         median=0.1646912129807167
tracking_ratio     median=0.7258725118706881  (achieved/commanded vx, walking holds; n=99)
schedule_completion_rate=0.96
terminations={'schedule_complete': 96, 'fall': 4}

by hold:
     stand: cmd 0.00 -> achieved 0.078 m/s | tripod median=0.5635788721069059 (n=100)
     creep: cmd 0.25 -> achieved 0.200 m/s | tripod median=0.6750314853566526 (n=99)
       low: cmd 0.35 -> achieved 0.228 m/s | tripod median=0.7559636930721 (n=96)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
