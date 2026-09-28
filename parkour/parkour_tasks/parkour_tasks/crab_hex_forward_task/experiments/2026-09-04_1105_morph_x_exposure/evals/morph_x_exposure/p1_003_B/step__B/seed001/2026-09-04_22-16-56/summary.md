```
=== crab-hex gait eval ===
scenario   : step__B  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-04_15-49-15/model_4999.pt
episodes   : 100  unscored: 1
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.28482636285667684  p25=0.2209285687664777  p75=0.3505496464817539
tippy_tap_fraction median=0.37718505123568413
slip_ratio         median=0.5304351798915904
tracking_ratio     median=0.6364779470377597  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.54
terminations={'schedule_complete': 54, 'fall': 46}

by hold:
     stand: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.0 (n=97)
     creep: cmd 0.25 -> achieved 0.195 m/s | tripod median=0.49900150768377116 (n=96)
       low: cmd 0.35 -> achieved 0.150 m/s | tripod median=0.39245805630247627 (n=65)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
