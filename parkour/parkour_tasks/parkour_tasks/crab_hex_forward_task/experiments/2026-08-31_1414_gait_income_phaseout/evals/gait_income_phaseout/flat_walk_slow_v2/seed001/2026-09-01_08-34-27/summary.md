```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-01_02-22-26/model_9998.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5820232231307417  p25=0.5504103562939114  p75=0.6169609300832387
tippy_tap_fraction median=0.23739081431389122
slip_ratio         median=0.21836936774517526
tracking_ratio     median=0.44272871457353374  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}

by hold:
     stand: cmd 0.00 -> achieved 0.040 m/s | tripod median=0.5365148908678385 (n=100)
     creep: cmd 0.25 -> achieved 0.114 m/s | tripod median=0.6381167487215654 (n=100)
       low: cmd 0.35 -> achieved 0.149 m/s | tripod median=0.5863368868394582 (n=100)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
