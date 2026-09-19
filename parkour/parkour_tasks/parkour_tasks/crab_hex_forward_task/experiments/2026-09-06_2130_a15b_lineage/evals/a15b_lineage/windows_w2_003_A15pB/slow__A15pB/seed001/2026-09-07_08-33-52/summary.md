```
=== crab-hex gait eval ===
scenario   : slow__A15pB  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-07_02-19-49/model_14997.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5069299484728685  p25=0.4544596437548375  p75=0.5339747557804009
tippy_tap_fraction median=0.2722881750465549
slip_ratio         median=0.2253216954707874
tracking_ratio     median=0.5886241939820074  (achieved/commanded vx, walking holds; n=97)
schedule_completion_rate=0.85
terminations={'fall': 15, 'schedule_complete': 85}

by hold:
     stand: cmd 0.00 -> achieved 0.108 m/s | tripod median=0.5367493403226212 (n=100)
     creep: cmd 0.25 -> achieved 0.170 m/s | tripod median=0.5319883126243551 (n=97)
       low: cmd 0.35 -> achieved 0.176 m/s | tripod median=0.4424872610227616 (n=89)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
