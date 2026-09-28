```
=== crab-hex gait eval ===
scenario   : flat_walk_forward  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/2026-08-10_0058_tripod_stability/a5_pitch_w-0.1/logs/rsl_rl/crab_hex_flat_walk/2026-08-10_02-03-09/model_20998.pt
episodes   : 10  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.39299684087156644  p25=0.38376168589265314  p75=0.39810849004449683
tippy_tap_fraction median=0.07002856066788024
slip_ratio         median=0.023939744096047398
schedule_completion_rate=1.0
terminations={'schedule_complete': 10}

by hold:
       low: tripod median=0.4892825890555531 (n=10)
       mid: tripod median=0.40947920237806934 (n=10)
      high: tripod median=0.2759921335654545 (n=10)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
