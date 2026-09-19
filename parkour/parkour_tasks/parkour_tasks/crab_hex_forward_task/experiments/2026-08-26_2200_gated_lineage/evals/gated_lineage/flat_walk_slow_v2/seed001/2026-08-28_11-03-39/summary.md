```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-28_05-35-39/model_4998.pt
episodes   : 100  unscored: 1
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.26365030935716294  p25=0.237673134322921  p75=0.2919405124516131
tippy_tap_fraction median=0.23798286564407733
slip_ratio         median=0.4086389578377927
tracking_ratio     median=0.1544864047889924  (achieved/commanded vx, walking holds; n=99)
schedule_completion_rate=0.97
terminations={'schedule_complete': 97, 'fall': 3}

by hold:
     stand: cmd 0.00 -> achieved 0.012 m/s | tripod median=0.17441774422026757 (n=99)
     creep: cmd 0.25 -> achieved 0.032 m/s | tripod median=0.26289624990835764 (n=99)
       low: cmd 0.35 -> achieved 0.063 m/s | tripod median=0.3633536081063592 (n=99)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
