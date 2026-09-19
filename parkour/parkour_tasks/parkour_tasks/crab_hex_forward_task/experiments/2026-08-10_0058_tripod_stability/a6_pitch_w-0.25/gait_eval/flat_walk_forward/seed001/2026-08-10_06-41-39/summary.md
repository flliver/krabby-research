```
=== crab-hex gait eval ===
scenario   : flat_walk_forward  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/2026-08-10_0058_tripod_stability/a6_pitch_w-0.25/logs/rsl_rl/crab_hex_flat_walk/2026-08-10_02-23-21/model_20998.pt
episodes   : 10  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.40273336255449477  p25=0.39711104158076205  p75=0.4080651855915378
tippy_tap_fraction median=0.07143072789300486
slip_ratio         median=0.02086436109077909
schedule_completion_rate=1.0
terminations={'schedule_complete': 10}

by hold:
       low: tripod median=0.49309329129205576 (n=10)
       mid: tripod median=0.4214504875572097 (n=10)
      high: tripod median=0.27690501186715233 (n=10)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
