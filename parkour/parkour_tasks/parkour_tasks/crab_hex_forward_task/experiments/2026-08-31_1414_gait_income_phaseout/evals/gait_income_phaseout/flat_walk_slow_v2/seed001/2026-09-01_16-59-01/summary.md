```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-01_10-45-00/model_14997.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5927413006131914  p25=0.5547815206071439  p75=0.6131767057599389
tippy_tap_fraction median=0.25
slip_ratio         median=0.20280932412519143
tracking_ratio     median=0.48653388993010005  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.96
terminations={'schedule_complete': 96, 'fall': 4}

by hold:
     stand: cmd 0.00 -> achieved 0.055 m/s | tripod median=0.5985101664414431 (n=100)
     creep: cmd 0.25 -> achieved 0.125 m/s | tripod median=0.6226182163412317 (n=100)
       low: cmd 0.35 -> achieved 0.165 m/s | tripod median=0.5566785357135395 (n=100)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
