```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-03_15-41-21/model_24995.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.536231371813118  p25=0.4998474250342747  p75=0.5580670618977258
tippy_tap_fraction median=0.2456641604010025
slip_ratio         median=0.2261078033225148
tracking_ratio     median=0.42970201706613076  (achieved/commanded vx, walking holds; n=98)
schedule_completion_rate=0.96
terminations={'schedule_complete': 96, 'fall': 4}

by hold:
     stand: cmd 0.00 -> achieved 0.060 m/s | tripod median=0.594155805156421 (n=100)
     creep: cmd 0.25 -> achieved 0.108 m/s | tripod median=0.5413480924813208 (n=98)
       low: cmd 0.35 -> achieved 0.150 m/s | tripod median=0.48817112331437573 (n=98)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
