```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-03_20-42-17/model_24995.pt
episodes   : 100  unscored: 1
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5537509005913588  p25=0.5030975246870003  p75=0.5891447469043252
tippy_tap_fraction median=0.2757113821138212
slip_ratio         median=0.2530208438311792
tracking_ratio     median=0.5096864317031957  (achieved/commanded vx, walking holds; n=97)
schedule_completion_rate=0.51
terminations={'schedule_complete': 51, 'fall': 49}

by hold:
     stand: cmd 0.00 -> achieved 0.082 m/s | tripod median=0.6041150897979488 (n=99)
     creep: cmd 0.25 -> achieved 0.145 m/s | tripod median=0.5717499811069524 (n=96)
       low: cmd 0.35 -> achieved 0.148 m/s | tripod median=0.4167693139259199 (n=70)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
