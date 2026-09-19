```
=== crab-hex gait eval ===
scenario   : flat_walk_forward  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/2026-08-12_1550_seed_basin_search/seed8/logs/rsl_rl/crab_hex_flat_walk/2026-08-12_23-23-31/model_2999.pt
episodes   : 10  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.03702517140126451  p25=0.01871249926106275  p75=0.05949620237567987
tippy_tap_fraction median=0.2665176438468364
slip_ratio         median=0.1119364020191621
schedule_completion_rate=1.0
terminations={'schedule_complete': 10}

by hold:
       low: tripod median=0.028616044080000395 (n=10)
       mid: tripod median=0.04070893214460458 (n=10)
      high: tripod median=0.0 (n=10)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
