```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-04_02-59-06/model_24995.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.49477911057247254  p25=0.43321069313417027  p75=0.5523924097131334
tippy_tap_fraction median=0.2365024288688411
slip_ratio         median=0.23024696770694963
tracking_ratio     median=0.46159216313200235  (achieved/commanded vx, walking holds; n=97)
schedule_completion_rate=0.36
terminations={'schedule_complete': 36, 'fall': 64}

by hold:
     stand: cmd 0.00 -> achieved 0.057 m/s | tripod median=0.5972010557854803 (n=100)
     creep: cmd 0.25 -> achieved 0.122 m/s | tripod median=0.49210274648048613 (n=97)
       low: cmd 0.35 -> achieved 0.149 m/s | tripod median=0.35960140222347053 (n=73)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
