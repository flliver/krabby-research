```
=== crab-hex gait eval ===
scenario   : step__base  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-31_03-42-16/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5528021877332817  p25=0.49752761323007094  p75=0.5971257621671701
tippy_tap_fraction median=0.29812942953410015
slip_ratio         median=0.32067310192959597
tracking_ratio     median=0.3983475198235319  (achieved/commanded vx, walking holds; n=98)
schedule_completion_rate=0.61
terminations={'schedule_complete': 61, 'fall': 39}

by hold:
     stand: cmd 0.00 -> achieved 0.042 m/s | tripod median=0.5317777261577732 (n=100)
     creep: cmd 0.25 -> achieved 0.110 m/s | tripod median=0.6008161685804589 (n=98)
       low: cmd 0.35 -> achieved 0.123 m/s | tripod median=0.49348964406462337 (n=86)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
