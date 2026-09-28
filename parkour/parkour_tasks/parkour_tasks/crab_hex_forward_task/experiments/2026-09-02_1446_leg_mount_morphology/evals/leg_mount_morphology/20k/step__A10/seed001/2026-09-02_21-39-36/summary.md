```
=== crab-hex gait eval ===
scenario   : step__A10  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-31_10-46-19/model_19996.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.613575984930804  p25=0.5569951255931052  p75=0.664617857569924
tippy_tap_fraction median=0.23809523809523808
slip_ratio         median=0.20853098952320906
tracking_ratio     median=0.4266867965116253  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.94
terminations={'schedule_complete': 94, 'fall': 6}

by hold:
     stand: cmd 0.00 -> achieved 0.068 m/s | tripod median=0.6748794163987308 (n=100)
     creep: cmd 0.25 -> achieved 0.114 m/s | tripod median=0.6015681127199646 (n=100)
       low: cmd 0.35 -> achieved 0.128 m/s | tripod median=0.5745783851102634 (n=95)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
