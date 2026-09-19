```
=== crab-hex gait eval ===
scenario   : flat_walk_forward  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/sim_fine_tuning/2026-08-12_1305_stride_definancing/S2_power1_w0.05/logs/rsl_rl/crab_hex_flat_walk/2026-08-12_13-23-16/model_21998.pt
episodes   : 10  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.41976005003415917  p25=0.4155674882454861  p75=0.4298299094280216
tippy_tap_fraction median=0.07032828282828282
slip_ratio         median=0.023065874884732235
schedule_completion_rate=1.0
terminations={'schedule_complete': 10}

by hold:
       low: tripod median=0.5245259648302275 (n=10)
       mid: tripod median=0.4490916534495065 (n=10)
      high: tripod median=0.3041654001180706 (n=10)

[WARN] num_prop=75 but observation width is 1151; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
