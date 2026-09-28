```
=== crab-hex gait eval ===
scenario   : slow__A15pB  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-07_06-57-52/model_24995.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.3709449787411756  p25=0.32631108275100396  p75=0.4189303370983588
tippy_tap_fraction median=0.25649393029630596
slip_ratio         median=0.20560971078905096
tracking_ratio     median=0.6334926417869167  (achieved/commanded vx, walking holds; n=93)
schedule_completion_rate=0.69
terminations={'schedule_complete': 69, 'fall': 31}

by hold:
     stand: cmd 0.00 -> achieved 0.114 m/s | tripod median=0.33071196091763233 (n=100)
     creep: cmd 0.25 -> achieved 0.178 m/s | tripod median=0.4139744444801697 (n=93)
       low: cmd 0.35 -> achieved 0.182 m/s | tripod median=0.42273825173109625 (n=71)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
