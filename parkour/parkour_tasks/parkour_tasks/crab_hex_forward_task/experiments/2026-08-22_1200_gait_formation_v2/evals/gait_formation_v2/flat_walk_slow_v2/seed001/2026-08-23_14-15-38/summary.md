```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-23_08-22-05/model_9998.pt
episodes   : 100  unscored: 4
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5372040950825688  p25=0.5093263473688864  p75=0.576099246415179
tippy_tap_fraction median=0.23809523809523808
slip_ratio         median=0.2565925298822718
tracking_ratio     median=0.5036380011606357  (achieved/commanded vx, walking holds; n=95)
schedule_completion_rate=0.9
terminations={'schedule_complete': 90, 'fall': 10}

by hold:
     stand: cmd 0.00 -> achieved 0.019 m/s | tripod median=0.43361315822458135 (n=96)
     creep: cmd 0.25 -> achieved 0.145 m/s | tripod median=0.6447840556244618 (n=95)
       low: cmd 0.35 -> achieved 0.147 m/s | tripod median=0.5655558740495322 (n=92)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
