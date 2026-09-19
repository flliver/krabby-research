```
=== crab-hex gait eval ===
scenario   : step__A20pB  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-31_10-46-19/model_19996.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6289764200889525  p25=0.5768771431061773  p75=0.675289929629781
tippy_tap_fraction median=0.23635561762652063
slip_ratio         median=0.22488576470854357
tracking_ratio     median=0.3828685312757667  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}

by hold:
     stand: cmd 0.00 -> achieved 0.064 m/s | tripod median=0.6654893690047061 (n=100)
     creep: cmd 0.25 -> achieved 0.100 m/s | tripod median=0.6270778616345307 (n=100)
       low: cmd 0.35 -> achieved 0.126 m/s | tripod median=0.6119434033473621 (n=100)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
