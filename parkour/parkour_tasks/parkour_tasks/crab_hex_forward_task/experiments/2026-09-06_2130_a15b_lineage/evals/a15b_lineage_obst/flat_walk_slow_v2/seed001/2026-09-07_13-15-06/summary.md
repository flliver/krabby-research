```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-07_06-57-52/model_24995.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.3339606632566548  p25=0.2645103090704115  p75=0.3938118209129561
tippy_tap_fraction median=0.24077733860342557
slip_ratio         median=0.17492673448102686
tracking_ratio     median=0.5746255374969674  (achieved/commanded vx, walking holds; n=57)
schedule_completion_rate=0.29
terminations={'schedule_complete': 29, 'fall': 71}

by hold:
     stand: cmd 0.00 -> achieved 0.091 m/s | tripod median=0.2684792036459035 (n=100)
     creep: cmd 0.25 -> achieved 0.159 m/s | tripod median=0.4207753052877082 (n=56)
       low: cmd 0.35 -> achieved 0.161 m/s | tripod median=0.4645407438233128 (n=34)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
