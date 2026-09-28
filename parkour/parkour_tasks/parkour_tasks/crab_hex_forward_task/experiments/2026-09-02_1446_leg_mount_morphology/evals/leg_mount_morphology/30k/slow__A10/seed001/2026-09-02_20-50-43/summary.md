```
=== crab-hex gait eval ===
scenario   : slow__A10  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-02_00-42-53/model_29994.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6338567533262259  p25=0.6149884222413067  p75=0.6533717948167181
tippy_tap_fraction median=0.2443755169561621
slip_ratio         median=0.2394684354386694
tracking_ratio     median=0.38292915974395225  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}

by hold:
     stand: cmd 0.00 -> achieved 0.061 m/s | tripod median=0.7212348210731663 (n=100)
     creep: cmd 0.25 -> achieved 0.105 m/s | tripod median=0.6129701017649858 (n=100)
       low: cmd 0.35 -> achieved 0.119 m/s | tripod median=0.5836957222597379 (n=100)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
