```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-24_17-07-16/model_24995.pt
episodes   : 100  unscored: 4
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5347340184507581  p25=0.4905633782797904  p75=0.576962866342916
tippy_tap_fraction median=0.24695076602707122
slip_ratio         median=0.24898647675751628
tracking_ratio     median=0.5524858516779257  (achieved/commanded vx, walking holds; n=95)
schedule_completion_rate=0.44
terminations={'fall': 56, 'schedule_complete': 44}

by hold:
     stand: cmd 0.00 -> achieved 0.032 m/s | tripod median=0.535740143437064 (n=96)
     creep: cmd 0.25 -> achieved 0.141 m/s | tripod median=0.5219776873928137 (n=95)
       low: cmd 0.35 -> achieved 0.180 m/s | tripod median=0.5946046675124712 (n=55)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
