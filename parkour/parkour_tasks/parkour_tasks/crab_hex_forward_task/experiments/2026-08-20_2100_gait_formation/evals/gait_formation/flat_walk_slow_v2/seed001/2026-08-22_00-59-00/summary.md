```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : logs/rsl_rl/crab_hex_flat_walk/2026-08-21_20-37-00/model_999.pt
episodes   : 100  unscored: 6
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.0037561627250150133  p25=0.0  p75=0.014961632722100162
tippy_tap_fraction median=0.44388111888111886
slip_ratio         median=0.4945811837586189
tracking_ratio     median=0.35145906707088725  (achieved/commanded vx, walking holds; n=48)
schedule_completion_rate=0.0
terminations={'fall': 100}

by hold:
     stand: cmd 0.00 -> achieved 0.033 m/s | tripod median=0.0 (n=94)
     creep: cmd 0.25 -> achieved 0.088 m/s | tripod median=0.0 (n=47)
       low: cmd 0.35 -> achieved 0.051 m/s | tripod median=0.017893951491409855 (n=3)

[WARN] num_prop=75 but observation width is 1127; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
