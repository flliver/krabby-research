```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-04_05-18-20/model_24995.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5619581128933606  p25=0.5129008319168852  p75=0.6082285978767525
tippy_tap_fraction median=0.2721661054994388
slip_ratio         median=0.24803715214860106
tracking_ratio     median=0.5104840606872347  (achieved/commanded vx, walking holds; n=95)
schedule_completion_rate=0.27
terminations={'fall': 73, 'schedule_complete': 27}

by hold:
     stand: cmd 0.00 -> achieved 0.081 m/s | tripod median=0.6773442476683488 (n=100)
     creep: cmd 0.25 -> achieved 0.134 m/s | tripod median=0.4940002265862931 (n=95)
       low: cmd 0.35 -> achieved 0.140 m/s | tripod median=0.3565426683887765 (n=54)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
