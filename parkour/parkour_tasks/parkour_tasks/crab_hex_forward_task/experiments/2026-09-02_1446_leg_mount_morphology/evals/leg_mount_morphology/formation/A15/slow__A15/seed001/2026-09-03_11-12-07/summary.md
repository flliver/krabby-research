```
=== crab-hex gait eval ===
scenario   : slow__A15  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-03_04-53-36/model_4999.pt
episodes   : 100  unscored: 1
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.24285448695467526  p25=0.11009920633010684  p75=0.32147764810790785
tippy_tap_fraction median=0.398132134209616
slip_ratio         median=0.36914412484212467
tracking_ratio     median=0.5254677324743946  (achieved/commanded vx, walking holds; n=86)
schedule_completion_rate=0.46
terminations={'schedule_complete': 46, 'fall': 54}

by hold:
     stand: cmd 0.00 -> achieved 0.013 m/s | tripod median=0.012380393543989923 (n=99)
     creep: cmd 0.25 -> achieved 0.130 m/s | tripod median=0.3812085797100194 (n=86)
       low: cmd 0.35 -> achieved 0.189 m/s | tripod median=0.47757376460069306 (n=59)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
