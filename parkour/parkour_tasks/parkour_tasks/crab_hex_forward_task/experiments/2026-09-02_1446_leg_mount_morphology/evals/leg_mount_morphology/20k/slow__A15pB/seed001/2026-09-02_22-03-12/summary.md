```
=== crab-hex gait eval ===
scenario   : slow__A15pB  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-31_10-46-19/model_19996.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6753735846667248  p25=0.6495831395875875  p75=0.69402777141761
tippy_tap_fraction median=0.2266317615315431
slip_ratio         median=0.21475329896489953
tracking_ratio     median=0.39345935976146973  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}

by hold:
     stand: cmd 0.00 -> achieved 0.056 m/s | tripod median=0.6754355456362622 (n=100)
     creep: cmd 0.25 -> achieved 0.095 m/s | tripod median=0.6685465256650265 (n=100)
       low: cmd 0.35 -> achieved 0.143 m/s | tripod median=0.6937818761188097 (n=100)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
