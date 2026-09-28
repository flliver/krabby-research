```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-06_23-59-54/model_9998.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.428660575283233  p25=0.39278504633923345  p75=0.47423435417581195
tippy_tap_fraction median=0.2508064516129032
slip_ratio         median=0.22424443294657273
tracking_ratio     median=0.6387304325715382  (achieved/commanded vx, walking holds; n=99)
schedule_completion_rate=0.56
terminations={'schedule_complete': 56, 'fall': 44}

by hold:
     stand: cmd 0.00 -> achieved 0.090 m/s | tripod median=0.3682439526816996 (n=100)
     creep: cmd 0.25 -> achieved 0.175 m/s | tripod median=0.47949492103748614 (n=99)
       low: cmd 0.35 -> achieved 0.189 m/s | tripod median=0.4919643706530295 (n=77)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
