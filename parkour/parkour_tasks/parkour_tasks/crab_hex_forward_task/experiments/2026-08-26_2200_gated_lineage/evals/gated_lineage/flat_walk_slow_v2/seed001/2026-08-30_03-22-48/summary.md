```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-29_22-02-49/model_9997.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.4340119119690862  p25=0.4036383262555967  p75=0.4636807710474333
tippy_tap_fraction median=0.296551724137931
slip_ratio         median=0.3135615964449863
tracking_ratio     median=0.4868513159024892  (achieved/commanded vx, walking holds; n=98)
schedule_completion_rate=0.92
terminations={'schedule_complete': 92, 'fall': 8}

by hold:
     stand: cmd 0.00 -> achieved 0.026 m/s | tripod median=0.16046130455709853 (n=100)
     creep: cmd 0.25 -> achieved 0.152 m/s | tripod median=0.6068935566021407 (n=96)
       low: cmd 0.35 -> achieved 0.134 m/s | tripod median=0.5553088069982213 (n=92)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
