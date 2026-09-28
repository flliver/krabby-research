```
=== crab-hex gait eval ===
scenario   : flat_walk_forward  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/2026-08-12_1550_seed_basin_search/seed2/logs/rsl_rl/crab_hex_flat_walk/2026-08-12_17-19-28/model_2999.pt
episodes   : 10  unscored: 1
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.07825125464889712  p25=0.07334397455019379  p75=0.0992119249042543
tippy_tap_fraction median=0.23500417710944027
slip_ratio         median=0.09751466409601281
schedule_completion_rate=0.8
terminations={'schedule_complete': 8, 'fall': 2}

by hold:
       low: tripod median=0.12226751795339592 (n=9)
       mid: tripod median=0.10048308075750136 (n=9)
      high: tripod median=0.030266032552077132 (n=8)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
