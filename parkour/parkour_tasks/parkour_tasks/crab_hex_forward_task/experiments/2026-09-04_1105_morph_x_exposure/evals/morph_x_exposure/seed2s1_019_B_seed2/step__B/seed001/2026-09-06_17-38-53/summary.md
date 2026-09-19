```
=== crab-hex gait eval ===
scenario   : step__B  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-06_11-13-20/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.38185343608788325  p25=0.2982858523439841  p75=0.4409060510222505
tippy_tap_fraction median=0.23358908780903664
slip_ratio         median=0.22504768770521583
tracking_ratio     median=0.6196403048829915  (achieved/commanded vx, walking holds; n=74)
schedule_completion_rate=0.37
terminations={'fall': 63, 'schedule_complete': 37}

by hold:
     stand: cmd 0.00 -> achieved 0.067 m/s | tripod median=0.31911523881432957 (n=100)
     creep: cmd 0.25 -> achieved 0.178 m/s | tripod median=0.48864035914390447 (n=71)
       low: cmd 0.35 -> achieved 0.162 m/s | tripod median=0.4409815384363362 (n=41)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
