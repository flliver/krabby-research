```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-24_15-11-50/model_14997.pt
episodes   : 100  unscored: 4
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5731817499591441  p25=0.5376469302112774  p75=0.5969691444283611
tippy_tap_fraction median=0.2222222222222222
slip_ratio         median=0.24389257821077584
tracking_ratio     median=0.4673284813238666  (achieved/commanded vx, walking holds; n=96)
schedule_completion_rate=0.94
terminations={'schedule_complete': 94, 'fall': 6}

by hold:
     stand: cmd 0.00 -> achieved 0.022 m/s | tripod median=0.5011684247236491 (n=96)
     creep: cmd 0.25 -> achieved 0.139 m/s | tripod median=0.6327009788238216 (n=96)
       low: cmd 0.35 -> achieved 0.131 m/s | tripod median=0.5906048489457709 (n=94)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
