```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-23_08-22-05/model_9998.pt
episodes   : 100  unscored: 5
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.48631844732046753  p25=0.43148784974947085  p75=0.5350740901411971
tippy_tap_fraction median=0.249
slip_ratio         median=0.2681875074061198
tracking_ratio     median=0.4824796917135854  (achieved/commanded vx, walking holds; n=93)
schedule_completion_rate=0.49
terminations={'fall': 51, 'schedule_complete': 49}

by hold:
     stand: cmd 0.00 -> achieved 0.009 m/s | tripod median=0.324927189477724 (n=95)
     creep: cmd 0.25 -> achieved 0.131 m/s | tripod median=0.6309318028930476 (n=93)
       low: cmd 0.35 -> achieved 0.152 m/s | tripod median=0.5281022124267696 (n=76)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
