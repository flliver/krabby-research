```
=== crab-hex gait eval ===
scenario   : flat_walk_forward  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/2026-08-10_0058_tripod_stability/a7_angvel_w-0.05/logs/rsl_rl/crab_hex_flat_walk/2026-08-10_02-43-24/model_20998.pt
episodes   : 10  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.4123134891659122  p25=0.39382305934231604  p75=0.42557618695226596
tippy_tap_fraction median=0.06833849457680397
slip_ratio         median=0.02274502269682003
schedule_completion_rate=1.0
terminations={'schedule_complete': 10}

by hold:
       low: tripod median=0.5040986755170187 (n=10)
       mid: tripod median=0.4415493504534162 (n=10)
      high: tripod median=0.29862791587777626 (n=10)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
