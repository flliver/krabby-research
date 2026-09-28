```
=== crab-hex gait eval ===
scenario   : flat_walk_forward  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/short_runs/r10_reversal_w-0.15/logs/rsl_rl/crab_hex_flat_walk/2026-08-09_12-58-09/model_20998.pt
episodes   : 10  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.0  p25=0.0  p75=0.0
tippy_tap_fraction median=0.1948922963625365
slip_ratio         median=0.014833785908405958
schedule_completion_rate=0.9
terminations={'schedule_complete': 9, 'fall': 1}

by hold:
       low: tripod median=0.0 (n=10)
       mid: tripod median=0.0 (n=10)
      high: tripod median=0.0 (n=10)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
