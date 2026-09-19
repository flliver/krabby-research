```
=== crab-hex gait eval ===
scenario   : flat_walk_forward  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/2026-08-10_0058_tripod_stability/fromscratch_tripod_v4_swap_short/logs/rsl_rl/crab_hex_flat_walk/2026-08-11_22-23-46/model_2999.pt
episodes   : 10  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.2545953948588435  p25=0.2148192185627965  p75=0.2657557724275993
tippy_tap_fraction median=0.08679739679013401
slip_ratio         median=0.031361962003516784
schedule_completion_rate=1.0
terminations={'schedule_complete': 10}

by hold:
       low: tripod median=0.2832843495608539 (n=10)
       mid: tripod median=0.2504811819294632 (n=10)
      high: tripod median=0.21766126914299 (n=10)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
