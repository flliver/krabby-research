```
=== crab-hex gait eval ===
scenario   : flat_walk_forward  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/2026-08-13_1230_perpendicular_mounts/canary_ft/logs/rsl_rl/crab_hex_flat_walk/2026-08-13_11-51-09/model_21998.pt
episodes   : 10  unscored: 1
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.4199329396873935  p25=0.4142280682185157  p75=0.4255991429351602
tippy_tap_fraction median=0.07071425492478124
slip_ratio         median=0.027363021302440303
schedule_completion_rate=0.9
terminations={'fall': 1, 'schedule_complete': 9}

by hold:
       low: tripod median=0.3744212656428753 (n=9)
       mid: tripod median=0.45403836501101896 (n=9)
      high: tripod median=0.4375549230730738 (n=9)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
