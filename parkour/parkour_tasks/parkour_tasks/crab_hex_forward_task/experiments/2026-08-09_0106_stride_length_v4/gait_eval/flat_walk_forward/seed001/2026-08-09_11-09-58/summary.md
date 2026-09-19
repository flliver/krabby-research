```
=== crab-hex gait eval ===
scenario   : flat_walk_forward  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/stride_length_v4/logs/rsl_rl/crab_hex_flat_walk/2026-08-09_01-06-25/model_19999.pt
episodes   : 10  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.024194077903876884  p25=0.02025468650604026  p75=0.03045593022944403
tippy_tap_fraction median=0.3916169031227416
slip_ratio         median=0.017833765222948716
schedule_completion_rate=1.0
terminations={'schedule_complete': 10}

by hold:
       low: tripod median=0.02879384853473968 (n=10)
       mid: tripod median=0.02880467027161649 (n=10)
      high: tripod median=0.01457280920785299 (n=10)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
