```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-02_09-51-23/model_29994.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.614583089596525  p25=0.5849250711850977  p75=0.6350349596653443
tippy_tap_fraction median=0.23076923076923078
slip_ratio         median=0.2053627141374158
tracking_ratio     median=0.4595594047196708  (achieved/commanded vx, walking holds; n=95)
schedule_completion_rate=0.73
terminations={'fall': 27, 'schedule_complete': 73}

by hold:
     stand: cmd 0.00 -> achieved 0.084 m/s | tripod median=0.6487180032157218 (n=100)
     creep: cmd 0.25 -> achieved 0.124 m/s | tripod median=0.5651294340926226 (n=95)
       low: cmd 0.35 -> achieved 0.146 m/s | tripod median=0.6158217893084967 (n=78)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
