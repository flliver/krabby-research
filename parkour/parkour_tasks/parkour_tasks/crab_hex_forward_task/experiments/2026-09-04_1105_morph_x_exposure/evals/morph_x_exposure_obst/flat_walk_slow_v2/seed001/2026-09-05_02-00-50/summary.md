```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-04_19-33-30/model_4999.pt
episodes   : 100  unscored: 3
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.3879503885103138  p25=0.3506587022682295  p75=0.42875323850059105
tippy_tap_fraction median=0.3198639455782313
slip_ratio         median=0.2138414076603411
tracking_ratio     median=0.70635847197427  (achieved/commanded vx, walking holds; n=83)
schedule_completion_rate=0.15
terminations={'fall': 85, 'schedule_complete': 15}

by hold:
     stand: cmd 0.00 -> achieved 0.099 m/s | tripod median=0.3957087198750083 (n=97)
     creep: cmd 0.25 -> achieved 0.198 m/s | tripod median=0.42752448730107323 (n=83)
       low: cmd 0.35 -> achieved 0.175 m/s | tripod median=0.22814909443694356 (n=35)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
