```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-31_14-31-54/model_9998.pt
episodes   : 100  unscored: 1
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5993661009149255  p25=0.5616418116334733  p75=0.6394815261414049
tippy_tap_fraction median=0.29414208558371413
slip_ratio         median=0.2979592452493711
tracking_ratio     median=0.4548432982634908  (achieved/commanded vx, walking holds; n=98)
schedule_completion_rate=0.48
terminations={'fall': 52, 'schedule_complete': 48}

by hold:
     stand: cmd 0.00 -> achieved 0.036 m/s | tripod median=0.6223047096396311 (n=99)
     creep: cmd 0.25 -> achieved 0.126 m/s | tripod median=0.7212664184163522 (n=98)
       low: cmd 0.35 -> achieved 0.145 m/s | tripod median=0.3973251605080965 (n=96)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
