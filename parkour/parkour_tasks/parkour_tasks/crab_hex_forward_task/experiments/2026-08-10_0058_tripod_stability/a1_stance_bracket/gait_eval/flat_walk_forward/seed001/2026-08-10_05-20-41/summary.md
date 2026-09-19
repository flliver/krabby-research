```
=== crab-hex gait eval ===
scenario   : flat_walk_forward  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/2026-08-10_0058_tripod_stability/a1_stance_bracket/logs/rsl_rl/crab_hex_flat_walk/2026-08-10_01-01-24/model_20998.pt
episodes   : 10  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.4044948199751346  p25=0.37795391443979875  p75=0.42086122153416555
tippy_tap_fraction median=0.07916666666666666
slip_ratio         median=0.024971219060859903
schedule_completion_rate=1.0
terminations={'schedule_complete': 10}

by hold:
       low: tripod median=0.5216108652529347 (n=10)
       mid: tripod median=0.4173058563936784 (n=10)
      high: tripod median=0.28613095228483165 (n=10)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
