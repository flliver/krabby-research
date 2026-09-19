```
=== crab-hex gait eval ===
scenario   : slow__B  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-05_21-37-35/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.3823834035627693  p25=0.3177584952331646  p75=0.4332190472083415
tippy_tap_fraction median=0.36173940745460337
slip_ratio         median=0.5552325074498885
tracking_ratio     median=0.6419873678277057  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.79
terminations={'schedule_complete': 79, 'fall': 21}

by hold:
     stand: cmd 0.00 -> achieved -0.005 m/s | tripod median=0.0 (n=98)
     creep: cmd 0.25 -> achieved 0.185 m/s | tripod median=0.480530761574233 (n=100)
       low: cmd 0.35 -> achieved 0.188 m/s | tripod median=0.6749466753384354 (n=81)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
