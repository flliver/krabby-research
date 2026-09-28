```
=== crab-hex gait eval ===
scenario   : flat_walk_forward  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/2026-08-13_0035_mirror_symmetry/teacher_stack/logs/rsl_rl/crab_hex_teacher/2026-08-13_10-14-52/model_22000.pt
episodes   : 10  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5254651985415092  p25=0.512236759499609  p75=0.5274969535145242
tippy_tap_fraction median=0.054859913442452245
slip_ratio         median=0.018350572987895677
schedule_completion_rate=1.0
terminations={'schedule_complete': 10}

by hold:
       low: tripod median=0.5620039600529548 (n=10)
       mid: tripod median=0.5305430073285796 (n=10)
      high: tripod median=0.46896997534480994 (n=10)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
