```
=== crab-hex gait eval ===
scenario   : step__A15  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-04_22-02-27/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.4483136095638172  p25=0.4126427652507866  p75=0.49838330495375843
tippy_tap_fraction median=0.23273809523809524
slip_ratio         median=0.20724634758661878
tracking_ratio     median=0.6381616900404201  (achieved/commanded vx, walking holds; n=93)
schedule_completion_rate=0.42
terminations={'schedule_complete': 42, 'fall': 58}

by hold:
     stand: cmd 0.00 -> achieved 0.044 m/s | tripod median=0.4335507075977356 (n=100)
     creep: cmd 0.25 -> achieved 0.184 m/s | tripod median=0.5388553252182694 (n=92)
       low: cmd 0.35 -> achieved 0.168 m/s | tripod median=0.3368049897025888 (n=61)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
