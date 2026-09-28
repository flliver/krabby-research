```
=== crab-hex gait eval ===
scenario   : flat_walk_forward  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/2026-08-12_1017_lean_reduction/L1_orientation_w-2.0/logs/rsl_rl/crab_hex_flat_walk/2026-08-12_10-18-02/model_21998.pt
episodes   : 10  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.3972880013484099  p25=0.3931227988597985  p75=0.40731924542300013
tippy_tap_fraction median=0.06564423170850835
slip_ratio         median=0.02208727072812347
schedule_completion_rate=1.0
terminations={'schedule_complete': 10}

by hold:
       low: tripod median=0.49709299500524096 (n=10)
       mid: tripod median=0.4284388444662197 (n=10)
      high: tripod median=0.2574092272137754 (n=10)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
