```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-24_09-19-09/model_14997.pt
episodes   : 100  unscored: 5
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5030943129525186  p25=0.45060621246097754  p75=0.5447690274117658
tippy_tap_fraction median=0.257389341875364
slip_ratio         median=0.2835295599418033
tracking_ratio     median=0.4624601809493691  (achieved/commanded vx, walking holds; n=95)
schedule_completion_rate=0.33
terminations={'fall': 67, 'schedule_complete': 33}

by hold:
     stand: cmd 0.00 -> achieved 0.034 m/s | tripod median=0.5685305444059698 (n=95)
     creep: cmd 0.25 -> achieved 0.139 m/s | tripod median=0.5568111513767564 (n=95)
       low: cmd 0.35 -> achieved 0.131 m/s | tripod median=0.311800139446346 (n=76)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
