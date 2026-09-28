```
=== crab-hex gait eval ===
scenario   : slow__A10  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-03_02-14-33/model_4999.pt
episodes   : 100  unscored: 2
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.1451437838996185  p25=0.07548492943584817  p75=0.24929392841549408
tippy_tap_fraction median=0.43746091307066914
slip_ratio         median=0.5537332434058058
tracking_ratio     median=0.6849977743037956  (achieved/commanded vx, walking holds; n=85)
schedule_completion_rate=0.23
terminations={'fall': 77, 'schedule_complete': 23}

by hold:
     stand: cmd 0.00 -> achieved -0.001 m/s | tripod median=0.0 (n=96)
     creep: cmd 0.25 -> achieved 0.187 m/s | tripod median=0.31257415461648463 (n=82)
       low: cmd 0.35 -> achieved 0.196 m/s | tripod median=0.4543905829328553 (n=27)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
