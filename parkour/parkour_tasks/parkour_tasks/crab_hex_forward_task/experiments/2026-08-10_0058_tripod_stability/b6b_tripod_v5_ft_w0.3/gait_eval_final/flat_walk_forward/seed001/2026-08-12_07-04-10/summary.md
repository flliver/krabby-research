```
=== crab-hex gait eval ===
scenario   : flat_walk_forward  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/2026-08-10_0058_tripod_stability/b6b_tripod_v5_ft_w0.3/logs/rsl_rl/crab_hex_flat_walk/2026-08-12_02-27-09/model_21998.pt
episodes   : 10  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.4032716850783367  p25=0.3955235705850272  p75=0.41252344389438617
tippy_tap_fraction median=0.07283427495291903
slip_ratio         median=0.02265896501882376
schedule_completion_rate=1.0
terminations={'schedule_complete': 10}

by hold:
       low: tripod median=0.4933076119827531 (n=10)
       mid: tripod median=0.4195557227096492 (n=10)
      high: tripod median=0.29820906592821306 (n=10)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
