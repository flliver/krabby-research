```
=== crab-hex gait eval ===
scenario   : fwd__A10pB  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-02_00-42-53/model_29994.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5226389838336252  p25=0.49444435343354587  p75=0.5422903554333359
tippy_tap_fraction median=0.29046043562172597
slip_ratio         median=0.2707966844016546
tracking_ratio     median=0.29036027918214014  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.76
terminations={'schedule_complete': 76, 'fall': 24}

by hold:
       low: cmd 0.30 -> achieved 0.114 m/s | tripod median=0.6060772054992218 (n=100)
       mid: cmd 0.47 -> achieved 0.125 m/s | tripod median=0.5225558681506196 (n=100)
      high: cmd 0.65 -> achieved 0.135 m/s | tripod median=0.43751193826024404 (n=98)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
