```
=== crab-hex gait eval ===
scenario   : flat_walk_forward  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/2026-08-10_0058_tripod_stability/a4_airtime_thresh0.10/logs/rsl_rl/crab_hex_flat_walk/2026-08-10_01-43-00/model_20998.pt
episodes   : 10  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.38266619811554353  p25=0.36064319279981766  p75=0.39022901809470917
tippy_tap_fraction median=0.08038897377132671
slip_ratio         median=0.022193494289847706
schedule_completion_rate=1.0
terminations={'schedule_complete': 10}

by hold:
       low: tripod median=0.4592191676822571 (n=10)
       mid: tripod median=0.39393443808052053 (n=10)
      high: tripod median=0.2929072513652712 (n=10)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
