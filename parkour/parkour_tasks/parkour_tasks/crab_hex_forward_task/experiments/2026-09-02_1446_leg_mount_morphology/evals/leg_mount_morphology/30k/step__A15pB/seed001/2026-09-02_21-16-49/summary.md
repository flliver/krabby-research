```
=== crab-hex gait eval ===
scenario   : step__A15pB  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-02_00-42-53/model_29994.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.600337418889302  p25=0.5543985423894775  p75=0.6423554017301807
tippy_tap_fraction median=0.265441512886991
slip_ratio         median=0.255361837863894
tracking_ratio     median=0.3646514826194057  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}

by hold:
     stand: cmd 0.00 -> achieved 0.064 m/s | tripod median=0.7131997618821395 (n=100)
     creep: cmd 0.25 -> achieved 0.103 m/s | tripod median=0.5828194931497555 (n=100)
       low: cmd 0.35 -> achieved 0.109 m/s | tripod median=0.538615428240161 (n=100)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
