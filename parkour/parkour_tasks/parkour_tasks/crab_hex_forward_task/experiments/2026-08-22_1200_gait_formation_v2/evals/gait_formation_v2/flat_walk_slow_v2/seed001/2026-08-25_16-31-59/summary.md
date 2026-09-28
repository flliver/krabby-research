```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-25_10-37-01/model_19996.pt
episodes   : 100  unscored: 5
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5924626193320314  p25=0.5643126983136653  p75=0.6237311779625019
tippy_tap_fraction median=0.21176470588235294
slip_ratio         median=0.22419619675368532
tracking_ratio     median=0.4544874776474457  (achieved/commanded vx, walking holds; n=95)
schedule_completion_rate=0.9
terminations={'schedule_complete': 90, 'fall': 10}

by hold:
     stand: cmd 0.00 -> achieved 0.040 m/s | tripod median=0.6007064833103479 (n=95)
     creep: cmd 0.25 -> achieved 0.123 m/s | tripod median=0.5885517487079741 (n=95)
       low: cmd 0.35 -> achieved 0.144 m/s | tripod median=0.6078601938826638 (n=92)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
