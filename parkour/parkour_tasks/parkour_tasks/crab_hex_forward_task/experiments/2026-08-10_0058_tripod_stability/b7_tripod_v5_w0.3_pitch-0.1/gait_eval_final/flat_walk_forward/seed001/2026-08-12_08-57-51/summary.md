```
=== crab-hex gait eval ===
scenario   : flat_walk_forward  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/2026-08-10_0058_tripod_stability/b7_tripod_v5_w0.3_pitch-0.1/logs/rsl_rl/crab_hex_flat_walk/2026-08-12_04-20-50/model_21998.pt
episodes   : 10  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.4181258685592313  p25=0.41134026384651856  p75=0.4250057639470081
tippy_tap_fraction median=0.06536205642477985
slip_ratio         median=0.022734319496064498
schedule_completion_rate=1.0
terminations={'schedule_complete': 10}

by hold:
       low: tripod median=0.5355386638101214 (n=10)
       mid: tripod median=0.4325333553562364 (n=10)
      high: tripod median=0.2882150801613617 (n=10)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
