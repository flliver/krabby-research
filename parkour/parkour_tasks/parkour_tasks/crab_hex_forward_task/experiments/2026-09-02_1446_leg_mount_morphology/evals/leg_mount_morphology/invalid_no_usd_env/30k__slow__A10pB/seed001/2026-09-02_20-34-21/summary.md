```
=== crab-hex gait eval ===
scenario   : slow__A10pB  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-02_00-42-53/model_29994.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5803721525155086  p25=0.5377032060552146  p75=0.6110182255217884
tippy_tap_fraction median=0.2406235349273324
slip_ratio         median=0.24035734789856478
tracking_ratio     median=0.41959658084164575  (achieved/commanded vx, walking holds; n=98)
schedule_completion_rate=0.81
terminations={'fall': 19, 'schedule_complete': 81}

by hold:
     stand: cmd 0.00 -> achieved 0.065 m/s | tripod median=0.7192357459497151 (n=100)
     creep: cmd 0.25 -> achieved 0.116 m/s | tripod median=0.5146828393275427 (n=98)
       low: cmd 0.35 -> achieved 0.129 m/s | tripod median=0.473947894452153 (n=86)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
