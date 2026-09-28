```
=== crab-hex gait eval ===
scenario   : flat_walk_forward  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/2026-08-12_1017_lean_reduction/L5_fwdprog_0.3/logs/rsl_rl/crab_hex_flat_walk/2026-08-12_11-35-05/model_21998.pt
episodes   : 10  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.418137597443763  p25=0.4094961209579168  p75=0.4298051500099738
tippy_tap_fraction median=0.06382293762575453
slip_ratio         median=0.022951887297947688
schedule_completion_rate=1.0
terminations={'schedule_complete': 10}

by hold:
       low: tripod median=0.5229986404499123 (n=10)
       mid: tripod median=0.4570672512136561 (n=10)
      high: tripod median=0.29289182890588195 (n=10)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
