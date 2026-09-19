```
=== crab-hex gait eval ===
scenario   : flat_walk_forward  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/2026-08-10_0058_tripod_stability/b1_tripod_reward_w0.15/logs/rsl_rl/crab_hex_flat_walk/2026-08-10_16-22-42/model_20998.pt
episodes   : 10  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.41148074270480506  p25=0.3758735852977711  p75=0.418970010525945
tippy_tap_fraction median=0.0782520325203252
slip_ratio         median=0.02438637565965037
schedule_completion_rate=1.0
terminations={'schedule_complete': 10}

by hold:
       low: tripod median=0.5092648377231017 (n=10)
       mid: tripod median=0.43549268241529193 (n=10)
      high: tripod median=0.28287925489610455 (n=10)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
