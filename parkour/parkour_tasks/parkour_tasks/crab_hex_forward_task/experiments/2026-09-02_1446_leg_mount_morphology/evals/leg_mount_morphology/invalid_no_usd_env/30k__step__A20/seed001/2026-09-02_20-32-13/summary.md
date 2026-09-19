```
=== crab-hex gait eval ===
scenario   : step__A20  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-02_00-42-53/model_29994.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6055297906849721  p25=0.55215363997279  p75=0.6577456475545017
tippy_tap_fraction median=0.25
slip_ratio         median=0.24121713216300766
tracking_ratio     median=0.5428296878957267  (achieved/commanded vx, walking holds; n=97)
schedule_completion_rate=0.26
terminations={'fall': 74, 'schedule_complete': 26}

by hold:
     stand: cmd 0.00 -> achieved 0.071 m/s | tripod median=0.7256757317740351 (n=100)
     creep: cmd 0.25 -> achieved 0.133 m/s | tripod median=0.49526409604054994 (n=97)
       low: cmd 0.35 -> achieved 0.122 m/s | tripod median=0.3815257310602551 (n=40)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
