```
=== crab-hex gait eval ===
scenario   : turn_walk_v1  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-31_01-12-23/model_19996.pt
episodes   : 100  unscored: 2
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6391818014720685  p25=0.6128545385816628  p75=0.6592848475900128
tippy_tap_fraction median=0.2583889927865349
slip_ratio         median=0.23436345696888838
tracking_ratio     median=0.5657217737402112  (achieved/commanded vx, walking holds; n=98)
schedule_completion_rate=0.92
terminations={'schedule_complete': 92, 'fall': 8}

by hold:
  straight: cmd 0.25 -> achieved 0.153 m/s | tripod median=0.6801981567227782 (n=98)
    turn_l: cmd 0.25 -> achieved 0.142 m/s | tripod median=0.6478590620030222 (n=92)
    turn_r: cmd 0.25 -> achieved 0.139 m/s | tripod median=0.620810897739072 (n=92)
  turn_hard: cmd 0.25 -> achieved 0.133 m/s | tripod median=0.6155510507172399 (n=92)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
