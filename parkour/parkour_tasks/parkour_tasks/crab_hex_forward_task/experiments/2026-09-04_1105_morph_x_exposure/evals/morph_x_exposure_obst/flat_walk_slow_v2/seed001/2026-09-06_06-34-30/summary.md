```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-06_00-09-00/model_9998.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.38892972704655693  p25=0.3447268898705064  p75=0.44703677090873817
tippy_tap_fraction median=0.34510334645669294
slip_ratio         median=0.5758260103286362
tracking_ratio     median=0.5781746602722339  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.83
terminations={'schedule_complete': 83, 'fall': 17}

by hold:
     stand: cmd 0.00 -> achieved 0.000 m/s | tripod median=0.0 (n=96)
     creep: cmd 0.25 -> achieved 0.166 m/s | tripod median=0.5904147172516414 (n=99)
       low: cmd 0.35 -> achieved 0.177 m/s | tripod median=0.6207280002144041 (n=85)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
