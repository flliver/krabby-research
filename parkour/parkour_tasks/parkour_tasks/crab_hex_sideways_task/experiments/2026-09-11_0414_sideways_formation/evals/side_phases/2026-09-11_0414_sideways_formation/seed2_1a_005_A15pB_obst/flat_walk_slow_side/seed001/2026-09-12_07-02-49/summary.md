```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_side  (Isaac-Crab-Hex-Side-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_side_flat_walk/2026-09-12_00-31-09/model_4999.pt
episodes   : 100  unscored: 4
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.35334718159135897  p25=0.2782564172804869  p75=0.4058180752617705
tippy_tap_fraction median=0.40007345085470086
slip_ratio         median=0.4326931463198578
tracking_ratio     median=0.5912996821920173  (achieved/commanded vx, walking holds; n=89)
schedule_completion_rate=0.76
terminations={'schedule_complete': 76, 'fall': 24}

by hold:
     stand: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.0 (n=94)
     creep: cmd 0.25 -> achieved 0.178 m/s | tripod median=0.5218216340590104 (n=89)
       low: cmd 0.35 -> achieved 0.161 m/s | tripod median=0.5268846967444462 (n=79)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
