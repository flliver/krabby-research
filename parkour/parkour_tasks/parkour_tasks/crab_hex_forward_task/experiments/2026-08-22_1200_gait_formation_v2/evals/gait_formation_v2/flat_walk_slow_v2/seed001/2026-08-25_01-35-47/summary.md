```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-24_19-06-17/model_14997.pt
episodes   : 100  unscored: 4
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5278481148634033  p25=0.48845248952440184  p75=0.5610322440724195
tippy_tap_fraction median=0.24074074074074073
slip_ratio         median=0.24860953962474144
tracking_ratio     median=0.4582924745391217  (achieved/commanded vx, walking holds; n=95)
schedule_completion_rate=0.91
terminations={'schedule_complete': 91, 'fall': 9}

by hold:
     stand: cmd 0.00 -> achieved 0.012 m/s | tripod median=0.32948961214656514 (n=96)
     creep: cmd 0.25 -> achieved 0.145 m/s | tripod median=0.667284269781139 (n=95)
       low: cmd 0.35 -> achieved 0.123 m/s | tripod median=0.5996667485525762 (n=92)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
