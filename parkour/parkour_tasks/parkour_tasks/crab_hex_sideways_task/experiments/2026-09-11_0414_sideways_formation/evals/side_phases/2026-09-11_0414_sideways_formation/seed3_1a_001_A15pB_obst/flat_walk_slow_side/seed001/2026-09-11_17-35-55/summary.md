```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_side  (Isaac-Crab-Hex-Side-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_side_flat_walk/2026-09-11_10-57-55/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.32382554535109764  p25=0.267553922820109  p75=0.3716854100362169
tippy_tap_fraction median=0.4153945469734943
slip_ratio         median=0.47624375394020535
tracking_ratio     median=0.5538172771619458  (achieved/commanded vx, walking holds; n=97)
schedule_completion_rate=0.78
terminations={'schedule_complete': 78, 'fall': 22}

by hold:
     stand: cmd 0.00 -> achieved 0.006 m/s | tripod median=0.007006865332594881 (n=99)
     creep: cmd 0.25 -> achieved 0.164 m/s | tripod median=0.4924397082738775 (n=97)
       low: cmd 0.35 -> achieved 0.161 m/s | tripod median=0.49167459629818344 (n=86)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
