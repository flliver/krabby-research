```
=== crab-hex gait eval ===
scenario   : flat_walk_forward  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/2026-08-10_0058_tripod_stability/b4_tripod_reward_w0.3_maxhold0.3/logs/rsl_rl/crab_hex_flat_walk/2026-08-10_17-44-53/model_20998.pt
episodes   : 10  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.4116015905172378  p25=0.4011409938244711  p75=0.42408502998488756
tippy_tap_fraction median=0.07243973107995771
slip_ratio         median=0.02313019454096793
schedule_completion_rate=1.0
terminations={'schedule_complete': 10}

by hold:
       low: tripod median=0.4943157029744847 (n=10)
       mid: tripod median=0.44940338126177487 (n=10)
      high: tripod median=0.3004939263407269 (n=10)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
