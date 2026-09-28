```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-05_19-18-29/model_9998.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5045980095971849  p25=0.45689582820051927  p75=0.5590639635557321
tippy_tap_fraction median=0.2228429546865301
slip_ratio         median=0.1932155597679914
tracking_ratio     median=0.5450599558241829  (achieved/commanded vx, walking holds; n=96)
schedule_completion_rate=0.73
terminations={'schedule_complete': 73, 'fall': 27}

by hold:
     stand: cmd 0.00 -> achieved 0.065 m/s | tripod median=0.5290487967055539 (n=100)
     creep: cmd 0.25 -> achieved 0.153 m/s | tripod median=0.5625025387606539 (n=96)
       low: cmd 0.35 -> achieved 0.173 m/s | tripod median=0.4735875810184569 (n=83)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
