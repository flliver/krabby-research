```
=== crab-hex gait eval ===
scenario   : adhoc  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/stride_length_on/logs/rsl_rl/crab_hex_flat_walk/2026-08-08_09-51-01/model_19999.pt
episodes   : 10  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.0  p25=0.0  p75=0.002865856672463798
tippy_tap_fraction median=0.36551437542325926
slip_ratio         median=0.02526991836368174
schedule_completion_rate=0.8
terminations={'schedule_complete': 8, 'fall': 2}

by hold:
       low: tripod median=0.0 (n=10)
       mid: tripod median=0.0 (n=10)
      high: tripod median=0.0 (n=9)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
