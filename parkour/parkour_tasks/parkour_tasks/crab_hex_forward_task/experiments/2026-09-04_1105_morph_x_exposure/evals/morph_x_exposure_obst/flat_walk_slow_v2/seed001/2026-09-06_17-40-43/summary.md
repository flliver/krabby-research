```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-06_11-13-20/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.3929455923935654  p25=0.2954033449461494  p75=0.4558334037178414
tippy_tap_fraction median=0.23707542457542458
slip_ratio         median=0.2135739317145759
tracking_ratio     median=0.5960996713064898  (achieved/commanded vx, walking holds; n=80)
schedule_completion_rate=0.29
terminations={'schedule_complete': 29, 'fall': 71}

by hold:
     stand: cmd 0.00 -> achieved 0.067 m/s | tripod median=0.316042191284238 (n=100)
     creep: cmd 0.25 -> achieved 0.166 m/s | tripod median=0.50155139754216 (n=80)
       low: cmd 0.35 -> achieved 0.178 m/s | tripod median=0.3890309545174669 (n=49)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
