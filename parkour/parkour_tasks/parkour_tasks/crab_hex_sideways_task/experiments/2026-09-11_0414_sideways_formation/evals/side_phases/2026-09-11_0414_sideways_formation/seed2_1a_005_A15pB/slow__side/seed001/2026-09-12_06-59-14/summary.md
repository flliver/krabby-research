```
=== crab-hex gait eval ===
scenario   : slow__side  (Isaac-Crab-Hex-Side-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_side_flat_walk/2026-09-12_00-31-09/model_4999.pt
episodes   : 100  unscored: 7
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.38143486707124497  p25=0.3276349708270312  p75=0.42760644878618276
tippy_tap_fraction median=0.404025055817415
slip_ratio         median=0.4493738334089258
tracking_ratio     median=0.6439594952852566  (achieved/commanded vx, walking holds; n=88)
schedule_completion_rate=0.79
terminations={'schedule_complete': 79, 'fall': 21}

by hold:
     stand: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.0 (n=90)
     creep: cmd 0.25 -> achieved 0.189 m/s | tripod median=0.5379716479964713 (n=88)
       low: cmd 0.35 -> achieved 0.186 m/s | tripod median=0.5955238545474049 (n=80)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
