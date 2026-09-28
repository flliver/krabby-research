```
=== crab-hex gait eval ===
scenario   : slow__A10  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-31_10-46-19/model_19996.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6560929137107745  p25=0.6235111720664593  p75=0.6769977636654019
tippy_tap_fraction median=0.23053392658509453
slip_ratio         median=0.19812678595847058
tracking_ratio     median=0.4162520580585852  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}

by hold:
     stand: cmd 0.00 -> achieved 0.059 m/s | tripod median=0.67098279890896 (n=100)
     creep: cmd 0.25 -> achieved 0.105 m/s | tripod median=0.641755532800033 (n=100)
       low: cmd 0.35 -> achieved 0.146 m/s | tripod median=0.6567117476969258 (n=100)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
