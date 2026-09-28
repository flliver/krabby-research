```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-01_15-10-31/model_19996.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5610204464334294  p25=0.5156466206346091  p75=0.5993754620196945
tippy_tap_fraction median=0.24927745664739884
slip_ratio         median=0.22708181177020736
tracking_ratio     median=0.4259540669852825  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.99
terminations={'schedule_complete': 99, 'fall': 1}

by hold:
     stand: cmd 0.00 -> achieved 0.052 m/s | tripod median=0.5721576655549032 (n=100)
     creep: cmd 0.25 -> achieved 0.108 m/s | tripod median=0.5741541508873227 (n=100)
       low: cmd 0.35 -> achieved 0.147 m/s | tripod median=0.5548838866945004 (n=99)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
