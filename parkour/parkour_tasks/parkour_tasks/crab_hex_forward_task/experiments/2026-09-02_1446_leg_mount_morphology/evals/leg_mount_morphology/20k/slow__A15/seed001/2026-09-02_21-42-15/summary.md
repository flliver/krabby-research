```
=== crab-hex gait eval ===
scenario   : slow__A15  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-31_10-46-19/model_19996.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6611682043920462  p25=0.6299347524409699  p75=0.6842970342234915
tippy_tap_fraction median=0.23249937217478653
slip_ratio         median=0.20260471291812698
tracking_ratio     median=0.39880788046355864  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}

by hold:
     stand: cmd 0.00 -> achieved 0.056 m/s | tripod median=0.6644287212564292 (n=100)
     creep: cmd 0.25 -> achieved 0.096 m/s | tripod median=0.6449491786633504 (n=100)
       low: cmd 0.35 -> achieved 0.148 m/s | tripod median=0.6607132916020249 (n=100)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
