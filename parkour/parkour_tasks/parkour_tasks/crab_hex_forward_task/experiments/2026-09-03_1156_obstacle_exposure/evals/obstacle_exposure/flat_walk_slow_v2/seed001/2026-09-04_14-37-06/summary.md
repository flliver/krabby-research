```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-04_08-18-59/model_24995.pt
episodes   : 100  unscored: 1
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5762832859563529  p25=0.5358265854891621  p75=0.6235003516096058
tippy_tap_fraction median=0.2578231292517007
slip_ratio         median=0.21548670961332642
tracking_ratio     median=0.5150344343812041  (achieved/commanded vx, walking holds; n=97)
schedule_completion_rate=0.92
terminations={'schedule_complete': 92, 'fall': 8}

by hold:
     stand: cmd 0.00 -> achieved 0.086 m/s | tripod median=0.6401925678944239 (n=99)
     creep: cmd 0.25 -> achieved 0.135 m/s | tripod median=0.6089558540006368 (n=97)
       low: cmd 0.35 -> achieved 0.169 m/s | tripod median=0.5148146964355517 (n=95)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
