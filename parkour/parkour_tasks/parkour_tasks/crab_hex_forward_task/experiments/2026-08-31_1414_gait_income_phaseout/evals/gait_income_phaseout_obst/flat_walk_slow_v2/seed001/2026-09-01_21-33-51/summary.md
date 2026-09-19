```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-01_15-10-31/model_19996.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5426259242204216  p25=0.48797220208770886  p75=0.5982228121076423
tippy_tap_fraction median=0.26589331384727805
slip_ratio         median=0.23183175525714791
tracking_ratio     median=0.4755467221431339  (achieved/commanded vx, walking holds; n=99)
schedule_completion_rate=0.6
terminations={'schedule_complete': 60, 'fall': 40}

by hold:
     stand: cmd 0.00 -> achieved 0.059 m/s | tripod median=0.5716231053279749 (n=100)
     creep: cmd 0.25 -> achieved 0.125 m/s | tripod median=0.5362102194916983 (n=99)
       low: cmd 0.35 -> achieved 0.145 m/s | tripod median=0.5294211847844192 (n=68)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
