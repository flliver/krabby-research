```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-04_02-59-06/model_24995.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.524383254687093  p25=0.4706062295217211  p75=0.5614037062755673
tippy_tap_fraction median=0.242915061014772
slip_ratio         median=0.23513174707671636
tracking_ratio     median=0.4357845362638089  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.79
terminations={'schedule_complete': 79, 'fall': 21}

by hold:
     stand: cmd 0.00 -> achieved 0.051 m/s | tripod median=0.600554372505712 (n=100)
     creep: cmd 0.25 -> achieved 0.112 m/s | tripod median=0.5192514567639235 (n=100)
       low: cmd 0.35 -> achieved 0.149 m/s | tripod median=0.4341844776846756 (n=88)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
