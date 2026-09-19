```
=== crab-hex gait eval ===
scenario   : slow__B  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-06_11-13-20/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.49213680983848174  p25=0.4282956497776554  p75=0.5480523525736737
tippy_tap_fraction median=0.22452448999704344
slip_ratio         median=0.226821248526571
tracking_ratio     median=0.6639738890709199  (achieved/commanded vx, walking holds; n=90)
schedule_completion_rate=0.78
terminations={'schedule_complete': 78, 'fall': 22}

by hold:
     stand: cmd 0.00 -> achieved 0.068 m/s | tripod median=0.3062250659808732 (n=100)
     creep: cmd 0.25 -> achieved 0.185 m/s | tripod median=0.5788811635016717 (n=89)
       low: cmd 0.35 -> achieved 0.203 m/s | tripod median=0.6600705661480568 (n=79)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
