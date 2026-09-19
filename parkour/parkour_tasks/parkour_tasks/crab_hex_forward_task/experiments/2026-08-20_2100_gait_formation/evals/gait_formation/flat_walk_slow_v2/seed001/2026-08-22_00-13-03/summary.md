```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-21_19-51-06/model_999.pt
episodes   : 100  unscored: 4
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.00045268149380623654  p25=0.0  p75=0.004631535880863501
tippy_tap_fraction median=0.45819036726394713
slip_ratio         median=0.41536251096499255
tracking_ratio     median=0.3911564628206714  (achieved/commanded vx, walking holds; n=55)
schedule_completion_rate=0.01
terminations={'fall': 99, 'schedule_complete': 1}

by hold:
     stand: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.0 (n=96)
     creep: cmd 0.25 -> achieved 0.098 m/s | tripod median=0.0014950797292348928 (n=55)
       low: cmd 0.35 -> achieved 0.117 m/s | tripod median=0.0015054237616784474 (n=9)

[WARN] num_prop=75 but observation width is 1127; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
