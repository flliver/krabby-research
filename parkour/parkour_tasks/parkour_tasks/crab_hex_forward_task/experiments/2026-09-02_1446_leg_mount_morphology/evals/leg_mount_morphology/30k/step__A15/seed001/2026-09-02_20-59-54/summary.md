```
=== crab-hex gait eval ===
scenario   : step__A15  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-02_00-42-53/model_29994.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5972511789092191  p25=0.5558512900478316  p75=0.6443742847605956
tippy_tap_fraction median=0.26168478260869565
slip_ratio         median=0.2539531946714923
tracking_ratio     median=0.37976232997482584  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.94
terminations={'schedule_complete': 94, 'fall': 6}

by hold:
     stand: cmd 0.00 -> achieved 0.065 m/s | tripod median=0.7185147501044882 (n=100)
     creep: cmd 0.25 -> achieved 0.107 m/s | tripod median=0.5594585918948978 (n=100)
       low: cmd 0.35 -> achieved 0.113 m/s | tripod median=0.5482800308606096 (n=97)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
