```
=== crab-hex gait eval ===
scenario   : step__side  (Isaac-Crab-Hex-Side-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_side_teacher/2026-09-12_07-45-40/model_19996.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.31202904765011796  p25=0.24325215314681076  p75=0.4049047263803893
tippy_tap_fraction median=0.3646329277611238
slip_ratio         median=0.463342031301668
tracking_ratio     median=0.5366010815092828  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.7
terminations={'fall': 30, 'schedule_complete': 70}

by hold:
     stand: cmd 0.00 -> achieved 0.005 m/s | tripod median=0.0 (n=93)
     creep: cmd 0.25 -> achieved 0.182 m/s | tripod median=0.5702049583651512 (n=99)
       low: cmd 0.35 -> achieved 0.137 m/s | tripod median=0.39151227154994955 (n=84)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
