```
=== crab-hex gait eval ===
scenario   : slow__A15pB  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-02_00-42-53/model_29994.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6582280488598562  p25=0.6408444205934102  p75=0.6760997040078944
tippy_tap_fraction median=0.24607600987969086
slip_ratio         median=0.23315444153590015
tracking_ratio     median=0.36723812883200696  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}

by hold:
     stand: cmd 0.00 -> achieved 0.061 m/s | tripod median=0.7204661246198415 (n=100)
     creep: cmd 0.25 -> achieved 0.099 m/s | tripod median=0.6375544450738553 (n=100)
       low: cmd 0.35 -> achieved 0.116 m/s | tripod median=0.625301036691555 (n=100)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
