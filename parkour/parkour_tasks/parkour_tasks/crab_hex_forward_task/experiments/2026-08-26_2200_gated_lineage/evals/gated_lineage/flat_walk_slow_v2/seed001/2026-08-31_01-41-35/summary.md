```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-30_20-11-36/model_19995.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.42438438048118665  p25=0.4019321551414386  p75=0.44804953165598776
tippy_tap_fraction median=0.3044492544492544
slip_ratio         median=0.29689120728182405
tracking_ratio     median=0.5139557529800043  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}

by hold:
     stand: cmd 0.00 -> achieved 0.006 m/s | tripod median=0.0383599522977889 (n=99)
     creep: cmd 0.25 -> achieved 0.138 m/s | tripod median=0.608125985852826 (n=100)
       low: cmd 0.35 -> achieved 0.166 m/s | tripod median=0.6143370430916335 (n=100)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
