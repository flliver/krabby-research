```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-27_01-06-53/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.49361562053318864  p25=0.4562653868670705  p75=0.5145712393548534
tippy_tap_fraction median=0.2569593558282208
slip_ratio         median=0.29638926491861906
tracking_ratio     median=0.44929943160361957  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.97
terminations={'schedule_complete': 97, 'fall': 3}

by hold:
     stand: cmd 0.00 -> achieved 0.012 m/s | tripod median=0.28041095826305223 (n=100)
     creep: cmd 0.25 -> achieved 0.130 m/s | tripod median=0.5959868880077792 (n=100)
       low: cmd 0.35 -> achieved 0.135 m/s | tripod median=0.6113869292421458 (n=97)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
