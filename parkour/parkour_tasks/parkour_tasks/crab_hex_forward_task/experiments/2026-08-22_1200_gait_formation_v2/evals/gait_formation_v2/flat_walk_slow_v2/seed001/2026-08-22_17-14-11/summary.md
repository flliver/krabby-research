```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-22_11-22-17/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5405412698293799  p25=0.5037823476401654  p75=0.5642525514976047
tippy_tap_fraction median=0.287147493029846
slip_ratio         median=0.26078096911631
tracking_ratio     median=0.5641276700407225  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.99
terminations={'schedule_complete': 99, 'fall': 1}

by hold:
     stand: cmd 0.00 -> achieved 0.036 m/s | tripod median=0.38534989896392624 (n=100)
     creep: cmd 0.25 -> achieved 0.165 m/s | tripod median=0.6574309199106583 (n=100)
       low: cmd 0.35 -> achieved 0.163 m/s | tripod median=0.5768471596668664 (n=100)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
