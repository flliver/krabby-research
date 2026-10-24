```
=== crab-hex gait eval ===
scenario   : slow__A15pB  (Isaac-Crab-Hex-Student-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_student/2026-09-08_05-54-01/model_24995.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.4819363739782212  p25=0.4431603028762926  p75=0.508417604211933
tippy_tap_fraction median=0.2666813114393981
slip_ratio         median=0.25587947574489184
tracking_ratio     median=0.6523472792578413  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.84
terminations={'schedule_complete': 84, 'fall': 16}

by hold:
     stand: cmd 0.00 -> achieved 0.156 m/s | tripod median=0.5523685874138232 (n=100)
     creep: cmd 0.25 -> achieved 0.201 m/s | tripod median=0.484047782180507 (n=99)
       low: cmd 0.35 -> achieved 0.172 m/s | tripod median=0.3711692563094519 (n=91)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
