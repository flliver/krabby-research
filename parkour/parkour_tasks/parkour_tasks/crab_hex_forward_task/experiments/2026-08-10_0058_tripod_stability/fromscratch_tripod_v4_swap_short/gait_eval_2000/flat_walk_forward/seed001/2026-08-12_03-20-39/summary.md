```
=== crab-hex gait eval ===
scenario   : flat_walk_forward  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/2026-08-10_0058_tripod_stability/fromscratch_tripod_v4_swap_short/logs/rsl_rl/crab_hex_flat_walk/2026-08-11_22-23-46/model_2000.pt
episodes   : 10  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.23002009520553082  p25=0.21478370170299785  p75=0.2452793284349289
tippy_tap_fraction median=0.0893009768009768
slip_ratio         median=0.03404424336541697
schedule_completion_rate=1.0
terminations={'schedule_complete': 10}

by hold:
       low: tripod median=0.2815755894345058 (n=10)
       mid: tripod median=0.2538963571865175 (n=10)
      high: tripod median=0.16512067854793944 (n=10)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
