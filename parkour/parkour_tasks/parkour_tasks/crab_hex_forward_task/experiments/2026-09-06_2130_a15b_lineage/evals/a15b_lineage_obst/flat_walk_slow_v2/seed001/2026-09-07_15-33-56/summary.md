```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-07_09-16-39/model_29994.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.3571119938650409  p25=0.2810485743056166  p75=0.4476337862272828
tippy_tap_fraction median=0.24025316455696202
slip_ratio         median=0.20842762742280765
tracking_ratio     median=0.6421952098688297  (achieved/commanded vx, walking holds; n=95)
schedule_completion_rate=0.48
terminations={'schedule_complete': 48, 'fall': 52}

by hold:
     stand: cmd 0.00 -> achieved 0.108 m/s | tripod median=0.23097380250595273 (n=100)
     creep: cmd 0.25 -> achieved 0.179 m/s | tripod median=0.43577835443838003 (n=95)
       low: cmd 0.35 -> achieved 0.196 m/s | tripod median=0.5223879279579032 (n=67)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
