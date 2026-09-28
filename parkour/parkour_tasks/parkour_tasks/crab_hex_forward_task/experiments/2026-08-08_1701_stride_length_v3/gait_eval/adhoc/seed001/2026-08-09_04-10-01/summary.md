```
=== crab-hex gait eval ===
scenario   : adhoc  (Isaac-Crab-Hex-Teacher-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/stride_length_v3/logs/rsl_rl/crab_hex_teacher/2026-08-08_23-55-28/model_21900.pt
episodes   : 10  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.02497191294805281  p25=0.012978214773555836  p75=0.029869684216754684
tippy_tap_fraction median=0.2101678808995882
slip_ratio         median=0.029570919095023115
schedule_completion_rate=0.8
terminations={'schedule_complete': 8, 'fall': 2}

by hold:
       low: tripod median=0.007663922723874636 (n=10)
       mid: tripod median=0.02311518474175 (n=9)
      high: tripod median=0.031365416669322264 (n=8)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
