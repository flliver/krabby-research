```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-05_05-27-19/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5026442659700119  p25=0.4436099560037378  p75=0.5489426439414301
tippy_tap_fraction median=0.2980749454979228
slip_ratio         median=0.29943379246393753
tracking_ratio     median=0.6769394854643334  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.91
terminations={'schedule_complete': 91, 'fall': 9}

by hold:
     stand: cmd 0.00 -> achieved 0.027 m/s | tripod median=0.4301698931989789 (n=100)
     creep: cmd 0.25 -> achieved 0.196 m/s | tripod median=0.5492123424388632 (n=100)
       low: cmd 0.35 -> achieved 0.204 m/s | tripod median=0.518698305425844 (n=98)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
