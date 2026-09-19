```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-25_19-09-15/model_24900.pt
episodes   : 100  unscored: 5
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6035272307921866  p25=0.5646778521464453  p75=0.6402750526728473
tippy_tap_fraction median=0.21764705882352942
slip_ratio         median=0.2432328164485282
tracking_ratio     median=0.42824080197598036  (achieved/commanded vx, walking holds; n=95)
schedule_completion_rate=0.94
terminations={'schedule_complete': 94, 'fall': 6}

by hold:
     stand: cmd 0.00 -> achieved 0.035 m/s | tripod median=0.5676943462490809 (n=95)
     creep: cmd 0.25 -> achieved 0.117 m/s | tripod median=0.6144571919417947 (n=95)
       low: cmd 0.35 -> achieved 0.138 m/s | tripod median=0.6332219978273395 (n=94)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
