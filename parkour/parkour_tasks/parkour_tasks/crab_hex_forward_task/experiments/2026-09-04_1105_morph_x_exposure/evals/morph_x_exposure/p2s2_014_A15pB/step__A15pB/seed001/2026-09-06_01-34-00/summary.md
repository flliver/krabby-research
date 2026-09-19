```
=== crab-hex gait eval ===
scenario   : step__A15pB  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-05_19-18-29/model_9998.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5002630153156542  p25=0.4415897736694846  p75=0.5476778853557942
tippy_tap_fraction median=0.2303370786516854
slip_ratio         median=0.19552942539353063
tracking_ratio     median=0.5307893188915129  (achieved/commanded vx, walking holds; n=92)
schedule_completion_rate=0.77
terminations={'schedule_complete': 77, 'fall': 23}

by hold:
     stand: cmd 0.00 -> achieved 0.065 m/s | tripod median=0.5359331172070618 (n=100)
     creep: cmd 0.25 -> achieved 0.147 m/s | tripod median=0.49299697925390545 (n=91)
       low: cmd 0.35 -> achieved 0.157 m/s | tripod median=0.45619433342099075 (n=81)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
