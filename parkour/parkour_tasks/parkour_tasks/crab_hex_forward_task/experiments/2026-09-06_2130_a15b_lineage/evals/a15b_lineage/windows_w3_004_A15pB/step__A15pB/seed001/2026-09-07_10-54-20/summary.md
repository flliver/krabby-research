```
=== crab-hex gait eval ===
scenario   : step__A15pB  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-07_04-38-50/model_19996.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.47897503612077297  p25=0.41698962097505143  p75=0.5347145962772026
tippy_tap_fraction median=0.24834408036145106
slip_ratio         median=0.20086056761861218
tracking_ratio     median=0.5755654291994334  (achieved/commanded vx, walking holds; n=90)
schedule_completion_rate=0.68
terminations={'schedule_complete': 68, 'fall': 32}

by hold:
     stand: cmd 0.00 -> achieved 0.119 m/s | tripod median=0.4631211267746359 (n=100)
     creep: cmd 0.25 -> achieved 0.161 m/s | tripod median=0.48971169444098284 (n=90)
       low: cmd 0.35 -> achieved 0.172 m/s | tripod median=0.5576855596301791 (n=73)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
