```
=== crab-hex gait eval ===
scenario   : flat_walk_forward  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/gait_tuned/logs/rsl_rl/crab_hex_flat_walk/2026-08-09_15-27-14/model_19999.pt
episodes   : 10  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.40114411159388763  p25=0.38589831776694966  p75=0.42200332934256
tippy_tap_fraction median=0.07972630392939299
slip_ratio         median=0.02305429917191658
schedule_completion_rate=1.0
terminations={'schedule_complete': 10}

by hold:
       low: tripod median=0.5036478190613304 (n=10)
       mid: tripod median=0.42324853528604783 (n=10)
      high: tripod median=0.31728613842677655 (n=10)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
