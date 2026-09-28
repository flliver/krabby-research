```
=== crab-hex gait eval ===
scenario   : flat_walk_forward  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/2026-08-12_1305_stride_definancing/S2_power1_w0.05/logs/rsl_rl/crab_hex_flat_walk/2026-08-12_13-23-16/model_21000.pt
episodes   : 10  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.3946832995207473  p25=0.38007315874104797  p75=0.407046444411466
tippy_tap_fraction median=0.07779326364692218
slip_ratio         median=0.025011194373932108
schedule_completion_rate=1.0
terminations={'schedule_complete': 10}

by hold:
       low: tripod median=0.48728371720849994 (n=10)
       mid: tripod median=0.4183129248333811 (n=10)
      high: tripod median=0.297183362573209 (n=10)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
