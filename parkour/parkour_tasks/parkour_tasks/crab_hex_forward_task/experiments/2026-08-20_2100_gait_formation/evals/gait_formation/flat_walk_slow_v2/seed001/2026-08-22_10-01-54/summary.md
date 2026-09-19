```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-22_04-17-22/model_9998.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.0  p25=0.0  p75=0.0002260305429572071
tippy_tap_fraction median=0.35673605655930873
slip_ratio         median=0.4157097247206076
tracking_ratio     median=0.5781211107202789  (achieved/commanded vx, walking holds; n=86)
schedule_completion_rate=0.46
terminations={'schedule_complete': 46, 'fall': 54}

by hold:
     stand: cmd 0.00 -> achieved 0.028 m/s | tripod median=0.0 (n=100)
     creep: cmd 0.25 -> achieved 0.153 m/s | tripod median=0.0 (n=85)
       low: cmd 0.35 -> achieved 0.187 m/s | tripod median=0.0 (n=73)

[WARN] num_prop=75 but observation width is 1127; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
