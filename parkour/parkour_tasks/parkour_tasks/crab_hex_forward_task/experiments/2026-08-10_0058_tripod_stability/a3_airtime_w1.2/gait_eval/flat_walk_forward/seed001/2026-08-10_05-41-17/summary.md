```
=== crab-hex gait eval ===
scenario   : flat_walk_forward  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/2026-08-10_0058_tripod_stability/a3_airtime_w1.2/logs/rsl_rl/crab_hex_flat_walk/2026-08-10_01-22-19/model_20998.pt
episodes   : 10  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.3887317718084531  p25=0.3865702608708803  p75=0.3971480623845903
tippy_tap_fraction median=0.0735360089722022
slip_ratio         median=0.022073845863123536
schedule_completion_rate=1.0
terminations={'schedule_complete': 10}

by hold:
       low: tripod median=0.49775028955217626 (n=10)
       mid: tripod median=0.4077830693359216 (n=10)
      high: tripod median=0.27542905198630363 (n=10)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
