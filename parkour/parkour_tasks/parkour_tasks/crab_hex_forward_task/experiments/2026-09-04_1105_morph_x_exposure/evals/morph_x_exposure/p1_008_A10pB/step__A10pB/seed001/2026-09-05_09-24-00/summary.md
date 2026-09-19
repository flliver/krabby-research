```
=== crab-hex gait eval ===
scenario   : step__A10pB  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-05_03-00-33/model_4999.pt
episodes   : 100  unscored: 4
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.3794688239359115  p25=0.30192999621659306  p75=0.42982951910088213
tippy_tap_fraction median=0.31458086590340834
slip_ratio         median=0.23432859192305416
tracking_ratio     median=0.6105849876764438  (achieved/commanded vx, walking holds; n=63)
schedule_completion_rate=0.21
terminations={'fall': 79, 'schedule_complete': 21}

by hold:
     stand: cmd 0.00 -> achieved 0.041 m/s | tripod median=0.3507885863895719 (n=96)
     creep: cmd 0.25 -> achieved 0.184 m/s | tripod median=0.47215212828922426 (n=60)
       low: cmd 0.35 -> achieved 0.129 m/s | tripod median=0.24374487397623518 (n=32)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
