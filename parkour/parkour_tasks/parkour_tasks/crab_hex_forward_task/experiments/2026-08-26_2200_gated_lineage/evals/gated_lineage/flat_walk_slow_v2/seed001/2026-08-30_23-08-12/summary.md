```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-30_18-06-13/model_16996.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.4328394556398676  p25=0.40329291636069575  p75=0.4596446106106584
tippy_tap_fraction median=0.28811566131710015
slip_ratio         median=0.2826263954294853
tracking_ratio     median=0.525188910829115  (achieved/commanded vx, walking holds; n=99)
schedule_completion_rate=0.98
terminations={'schedule_complete': 98, 'fall': 2}

by hold:
     stand: cmd 0.00 -> achieved 0.012 m/s | tripod median=0.11908728847187372 (n=100)
     creep: cmd 0.25 -> achieved 0.151 m/s | tripod median=0.6272375659908909 (n=99)
       low: cmd 0.35 -> achieved 0.158 m/s | tripod median=0.554684027538028 (n=98)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
