```
=== crab-hex gait eval ===
scenario   : step__A20pB  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-05_07-56-31/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.4105537764618085  p25=0.328579959075857  p75=0.4927721780778994
tippy_tap_fraction median=0.3376639811026279
slip_ratio         median=0.2961045439569795
tracking_ratio     median=0.6135813218147995  (achieved/commanded vx, walking holds; n=97)
schedule_completion_rate=0.54
terminations={'fall': 46, 'schedule_complete': 54}

by hold:
     stand: cmd 0.00 -> achieved 0.089 m/s | tripod median=0.32606458336166994 (n=100)
     creep: cmd 0.25 -> achieved 0.196 m/s | tripod median=0.5010121498932559 (n=95)
       low: cmd 0.35 -> achieved 0.137 m/s | tripod median=0.5120946506913202 (n=68)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
