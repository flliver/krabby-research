```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-30_06-22-28/model_11997.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.4314069944562717  p25=0.4173161046887852  p75=0.4446327747243599
tippy_tap_fraction median=0.2772167487684729
slip_ratio         median=0.28479545528519445
tracking_ratio     median=0.4639890920324603  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}

by hold:
     stand: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.07042426202723734 (n=100)
     creep: cmd 0.25 -> achieved 0.128 m/s | tripod median=0.5962238711618101 (n=100)
       low: cmd 0.35 -> achieved 0.141 m/s | tripod median=0.6366889192730439 (n=100)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
