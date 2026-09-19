```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-01_17-50-56/model_21995.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5761807161583132  p25=0.5368290080504925  p75=0.6046794290717318
tippy_tap_fraction median=0.24411683277962348
slip_ratio         median=0.20126963544165555
tracking_ratio     median=0.4526820718194845  (achieved/commanded vx, walking holds; n=97)
schedule_completion_rate=0.81
terminations={'schedule_complete': 81, 'fall': 19}

by hold:
     stand: cmd 0.00 -> achieved 0.062 m/s | tripod median=0.6302078577249046 (n=100)
     creep: cmd 0.25 -> achieved 0.121 m/s | tripod median=0.571536789918119 (n=97)
       low: cmd 0.35 -> achieved 0.149 m/s | tripod median=0.538150057134243 (n=84)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
