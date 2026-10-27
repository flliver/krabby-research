```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_side  (Isaac-Crab-Hex-Side-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_side_teacher/2026-09-12_07-45-40/model_19996.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.4066090453715135  p25=0.3807554160019531  p75=0.45294651945317754
tippy_tap_fraction median=0.3626692456479691
slip_ratio         median=0.4148059538714659
tracking_ratio     median=0.657663040124777  (achieved/commanded vx, walking holds; n=98)
schedule_completion_rate=0.8
terminations={'schedule_complete': 80, 'fall': 20}

by hold:
     stand: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.0 (n=91)
     creep: cmd 0.25 -> achieved 0.195 m/s | tripod median=0.6374675196043365 (n=98)
       low: cmd 0.35 -> achieved 0.194 m/s | tripod median=0.5989494541234954 (n=93)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
