```
=== crab-hex gait eval ===
scenario   : flat_walk_forward  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/2026-08-10_0058_tripod_stability/b6_tripod_v5_ft_w0.15/logs/rsl_rl/crab_hex_flat_walk/2026-08-12_01-47-04/model_21998.pt
episodes   : 10  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.39982980374881305  p25=0.3912550989524786  p75=0.4214312627710868
tippy_tap_fraction median=0.07029641827912722
slip_ratio         median=0.023048569804312892
schedule_completion_rate=1.0
terminations={'schedule_complete': 10}

by hold:
       low: tripod median=0.4982089552399951 (n=10)
       mid: tripod median=0.4229050761468938 (n=10)
      high: tripod median=0.3092523171370575 (n=10)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
