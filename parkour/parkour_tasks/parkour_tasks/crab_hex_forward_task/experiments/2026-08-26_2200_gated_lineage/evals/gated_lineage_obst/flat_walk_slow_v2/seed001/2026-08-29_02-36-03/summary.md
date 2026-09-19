```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-27_01-06-53/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.44718041342887804  p25=0.3962543236875035  p75=0.5009634041089612
tippy_tap_fraction median=0.260465789691404
slip_ratio         median=0.303090056017555
tracking_ratio     median=0.4400944220722224  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.68
terminations={'schedule_complete': 68, 'fall': 32}

by hold:
     stand: cmd 0.00 -> achieved 0.012 m/s | tripod median=0.28783250369423763 (n=100)
     creep: cmd 0.25 -> achieved 0.124 m/s | tripod median=0.5936728212673328 (n=100)
       low: cmd 0.35 -> achieved 0.127 m/s | tripod median=0.49194754307735483 (n=81)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
