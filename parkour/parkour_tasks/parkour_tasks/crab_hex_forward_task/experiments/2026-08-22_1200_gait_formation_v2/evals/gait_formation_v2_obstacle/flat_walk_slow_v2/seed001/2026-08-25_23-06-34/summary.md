```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-25_16-34-38/model_19996.pt
episodes   : 100  unscored: 7
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5615329978179805  p25=0.5147856609846256  p75=0.6044559051641687
tippy_tap_fraction median=0.21153846153846154
slip_ratio         median=0.2056772295348982
tracking_ratio     median=0.46194972027362124  (achieved/commanded vx, walking holds; n=92)
schedule_completion_rate=0.45
terminations={'schedule_complete': 45, 'fall': 55}

by hold:
     stand: cmd 0.00 -> achieved 0.030 m/s | tripod median=0.5806325004196894 (n=93)
     creep: cmd 0.25 -> achieved 0.133 m/s | tripod median=0.5881135821565978 (n=92)
       low: cmd 0.35 -> achieved 0.118 m/s | tripod median=0.4717829959075057 (n=55)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
