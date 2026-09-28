```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-30_03-51-02/model_9998.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.4316446072331105  p25=0.4149714664206594  p75=0.4523190278742579
tippy_tap_fraction median=0.2684182015167931
slip_ratio         median=0.274313710624623
tracking_ratio     median=0.45639219217335697  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.99
terminations={'schedule_complete': 99, 'fall': 1}

by hold:
     stand: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.07306527336562207 (n=100)
     creep: cmd 0.25 -> achieved 0.128 m/s | tripod median=0.5876021431342462 (n=100)
       low: cmd 0.35 -> achieved 0.142 m/s | tripod median=0.6290980059562474 (n=99)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
