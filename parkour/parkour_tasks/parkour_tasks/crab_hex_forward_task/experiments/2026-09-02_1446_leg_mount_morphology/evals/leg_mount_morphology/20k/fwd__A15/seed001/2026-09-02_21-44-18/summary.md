```
=== crab-hex gait eval ===
scenario   : fwd__A15  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-31_10-46-19/model_19996.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5568030361030013  p25=0.5267641926707782  p75=0.5759294344392569
tippy_tap_fraction median=0.2892324013033265
slip_ratio         median=0.25319825959969255
tracking_ratio     median=0.3094279258637075  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.99
terminations={'schedule_complete': 99, 'fall': 1}

by hold:
       low: cmd 0.30 -> achieved 0.124 m/s | tripod median=0.6308873121299057 (n=100)
       mid: cmd 0.47 -> achieved 0.134 m/s | tripod median=0.5689002280379445 (n=100)
      high: cmd 0.65 -> achieved 0.151 m/s | tripod median=0.4679391440868571 (n=100)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
