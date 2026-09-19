```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-23_18-12-12/model_14997.pt
episodes   : 100  unscored: 4
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5386512955788811  p25=0.5095302702023213  p75=0.5674767549942853
tippy_tap_fraction median=0.22302158273381295
slip_ratio         median=0.22743586865901036
tracking_ratio     median=0.48009682665454734  (achieved/commanded vx, walking holds; n=96)
schedule_completion_rate=0.87
terminations={'schedule_complete': 87, 'fall': 13}

by hold:
     stand: cmd 0.00 -> achieved 0.035 m/s | tripod median=0.47296043501361784 (n=96)
     creep: cmd 0.25 -> achieved 0.145 m/s | tripod median=0.603486227938147 (n=96)
       low: cmd 0.35 -> achieved 0.132 m/s | tripod median=0.5645120600583232 (n=88)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
