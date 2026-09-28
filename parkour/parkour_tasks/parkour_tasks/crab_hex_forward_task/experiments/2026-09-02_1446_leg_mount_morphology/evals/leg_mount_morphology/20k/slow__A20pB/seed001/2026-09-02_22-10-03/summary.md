```
=== crab-hex gait eval ===
scenario   : slow__A20pB  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-31_10-46-19/model_19996.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.666025376769166  p25=0.6400148861700358  p75=0.6947958340574305
tippy_tap_fraction median=0.23752924177790008
slip_ratio         median=0.21443779588213943
tracking_ratio     median=0.3972582633728402  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}

by hold:
     stand: cmd 0.00 -> achieved 0.055 m/s | tripod median=0.6692578552580797 (n=100)
     creep: cmd 0.25 -> achieved 0.095 m/s | tripod median=0.6600875698691517 (n=100)
       low: cmd 0.35 -> achieved 0.143 m/s | tripod median=0.6806047700065114 (n=100)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
