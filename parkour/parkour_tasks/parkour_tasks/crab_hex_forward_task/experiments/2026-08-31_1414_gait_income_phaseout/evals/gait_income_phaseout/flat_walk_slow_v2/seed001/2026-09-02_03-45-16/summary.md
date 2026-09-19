```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-01_22-33-16/model_26994.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6150027937927212  p25=0.590516711375505  p75=0.6374798926135136
tippy_tap_fraction median=0.23979166666666668
slip_ratio         median=0.2244469445065146
tracking_ratio     median=0.43698526447265695  (achieved/commanded vx, walking holds; n=99)
schedule_completion_rate=0.91
terminations={'schedule_complete': 91, 'fall': 9}

by hold:
     stand: cmd 0.00 -> achieved 0.056 m/s | tripod median=0.6867080866983277 (n=100)
     creep: cmd 0.25 -> achieved 0.118 m/s | tripod median=0.6010738865553185 (n=99)
       low: cmd 0.35 -> achieved 0.141 m/s | tripod median=0.562235048981752 (n=92)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
