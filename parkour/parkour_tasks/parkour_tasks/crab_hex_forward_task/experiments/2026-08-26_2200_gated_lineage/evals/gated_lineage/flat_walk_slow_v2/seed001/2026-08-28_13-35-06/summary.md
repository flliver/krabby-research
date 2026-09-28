```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-28_08-05-07/model_4998.pt
episodes   : 100  unscored: 1
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.00013640016911328206  p25=0.0  p75=0.015461334143968792
tippy_tap_fraction median=0.41522988505747127
slip_ratio         median=0.726704175633237
tracking_ratio     median=0.18182258818154062  (achieved/commanded vx, walking holds; n=81)
schedule_completion_rate=0.05
terminations={'fall': 95, 'schedule_complete': 5}

by hold:
     stand: cmd 0.00 -> achieved 0.017 m/s | tripod median=0.0 (n=95)
     creep: cmd 0.25 -> achieved 0.045 m/s | tripod median=0.0 (n=76)
       low: cmd 0.35 -> achieved 0.011 m/s | tripod median=0.04856323980211563 (n=8)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
