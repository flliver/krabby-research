```
=== crab-hex gait eval ===
scenario   : flat_walk_forward  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/2026-08-12_1017_lean_reduction/L1_orientation_w-2.0/logs/rsl_rl/crab_hex_flat_walk/2026-08-12_10-18-02/model_21000.pt
episodes   : 10  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.3893539368329314  p25=0.3713320133837191  p75=0.39492906739716827
tippy_tap_fraction median=0.0771516502175528
slip_ratio         median=0.02497651398331948
schedule_completion_rate=1.0
terminations={'schedule_complete': 10}

by hold:
       low: tripod median=0.5127161834996949 (n=10)
       mid: tripod median=0.4256058173909713 (n=10)
      high: tripod median=0.230247726351444 (n=10)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
