```
=== crab-hex gait eval ===
scenario   : slow__A15  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-04_22-02-27/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.4944483634703395  p25=0.45484430000884335  p75=0.5413115074898299
tippy_tap_fraction median=0.24349818083816197
slip_ratio         median=0.20561876074947483
tracking_ratio     median=0.6481390942543523  (achieved/commanded vx, walking holds; n=97)
schedule_completion_rate=0.9
terminations={'schedule_complete': 90, 'fall': 10}

by hold:
     stand: cmd 0.00 -> achieved 0.061 m/s | tripod median=0.4643706375659358 (n=100)
     creep: cmd 0.25 -> achieved 0.194 m/s | tripod median=0.6138167177878772 (n=94)
       low: cmd 0.35 -> achieved 0.185 m/s | tripod median=0.450811795185876 (n=90)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
