```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-02_00-42-53/model_29994.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6024495573451589  p25=0.5649227177681619  p75=0.6528574282738389
tippy_tap_fraction median=0.25287965047001193
slip_ratio         median=0.23437607963816787
tracking_ratio     median=0.48969167096937466  (achieved/commanded vx, walking holds; n=97)
schedule_completion_rate=0.4
terminations={'fall': 60, 'schedule_complete': 40}

by hold:
     stand: cmd 0.00 -> achieved 0.068 m/s | tripod median=0.7149723196015695 (n=100)
     creep: cmd 0.25 -> achieved 0.131 m/s | tripod median=0.5495628724060264 (n=96)
       low: cmd 0.35 -> achieved 0.137 m/s | tripod median=0.4526706898512979 (n=54)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
