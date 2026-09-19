```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-27_23-18-14/model_1999.pt
episodes   : 100  unscored: 1
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.1902192418317793  p25=0.152113202209622  p75=0.2438176727905225
tippy_tap_fraction median=0.392779095178314
slip_ratio         median=0.38729855598484164
tracking_ratio     median=0.47554130196165967  (achieved/commanded vx, walking holds; n=91)
schedule_completion_rate=0.38
terminations={'fall': 62, 'schedule_complete': 38}

by hold:
     stand: cmd 0.00 -> achieved 0.011 m/s | tripod median=0.0325724292618015 (n=99)
     creep: cmd 0.25 -> achieved 0.136 m/s | tripod median=0.3354546001914877 (n=91)
       low: cmd 0.35 -> achieved 0.137 m/s | tripod median=0.2580572427772178 (n=63)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
