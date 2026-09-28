```
=== crab-hex gait eval ===
scenario   : step__A15  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-06_05-05-22/model_9998.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.47658136162546294  p25=0.4057854921450309  p75=0.5343441888425821
tippy_tap_fraction median=0.21429837937515517
slip_ratio         median=0.19506912573932267
tracking_ratio     median=0.5821420241175389  (achieved/commanded vx, walking holds; n=99)
schedule_completion_rate=0.74
terminations={'schedule_complete': 74, 'fall': 26}

by hold:
     stand: cmd 0.00 -> achieved 0.049 m/s | tripod median=0.3987201646768681 (n=100)
     creep: cmd 0.25 -> achieved 0.168 m/s | tripod median=0.5466100934512117 (n=99)
       low: cmd 0.35 -> achieved 0.156 m/s | tripod median=0.538377749074416 (n=79)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
