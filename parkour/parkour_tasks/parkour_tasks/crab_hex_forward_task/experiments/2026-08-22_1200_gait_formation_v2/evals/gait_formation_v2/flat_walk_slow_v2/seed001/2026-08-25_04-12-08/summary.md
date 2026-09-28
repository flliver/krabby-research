```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-24_21-37-19/model_4999.pt
episodes   : 100  unscored: 8
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.15163266477112075  p25=0.11917688167979382  p75=0.1851068891325026
tippy_tap_fraction median=0.41358024691358025
slip_ratio         median=0.6100792626380867
tracking_ratio     median=0.03196512798834535  (achieved/commanded vx, walking holds; n=89)
schedule_completion_rate=0.52
terminations={'schedule_complete': 52, 'fall': 48}

by hold:
     stand: cmd 0.00 -> achieved -0.011 m/s | tripod median=0.19191582625867706 (n=92)
     creep: cmd 0.25 -> achieved 0.003 m/s | tripod median=0.15025644836724517 (n=89)
       low: cmd 0.35 -> achieved 0.011 m/s | tripod median=0.08066828196626902 (n=72)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
