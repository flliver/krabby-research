```
=== crab-hex gait eval ===
scenario   : flat_walk_forward  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/2026-08-13_0035_mirror_symmetry/teacher_stack/logs/rsl_rl/crab_hex_teacher/2026-08-13_09-29-03/model_20197.pt
episodes   : 10  unscored: 1
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5185350321689852  p25=0.5043136141536587  p75=0.5203908931093747
tippy_tap_fraction median=0.06342840788518456
slip_ratio         median=0.023525764029146874
schedule_completion_rate=0.9
terminations={'schedule_complete': 9, 'fall': 1}

by hold:
       low: tripod median=0.54068581917055 (n=9)
       mid: tripod median=0.5455544068608883 (n=9)
      high: tripod median=0.4618904393854314 (n=9)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
