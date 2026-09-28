```
=== crab-hex gait eval ===
scenario   : slow__A15pB  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-06_23-59-54/model_9998.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5125834596825507  p25=0.45670306516716086  p75=0.56063423156508
tippy_tap_fraction median=0.25247311827956986
slip_ratio         median=0.23079761795865308
tracking_ratio     median=0.6411448654448932  (achieved/commanded vx, walking holds; n=97)
schedule_completion_rate=0.92
terminations={'schedule_complete': 92, 'fall': 8}

by hold:
     stand: cmd 0.00 -> achieved 0.100 m/s | tripod median=0.4618513034477325 (n=100)
     creep: cmd 0.25 -> achieved 0.189 m/s | tripod median=0.5711288211751145 (n=97)
       low: cmd 0.35 -> achieved 0.188 m/s | tripod median=0.530310985257873 (n=94)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
