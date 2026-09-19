```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-05_00-31-35/model_4999.pt
episodes   : 100  unscored: 1
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.4699760501861945  p25=0.3890624367867845  p75=0.5037638801845432
tippy_tap_fraction median=0.25
slip_ratio         median=0.21303226285003163
tracking_ratio     median=0.5716910720412076  (achieved/commanded vx, walking holds; n=65)
schedule_completion_rate=0.34
terminations={'schedule_complete': 34, 'fall': 66}

by hold:
     stand: cmd 0.00 -> achieved 0.026 m/s | tripod median=0.3941128646862702 (n=99)
     creep: cmd 0.25 -> achieved 0.169 m/s | tripod median=0.5926103762303332 (n=64)
       low: cmd 0.35 -> achieved 0.160 m/s | tripod median=0.5029461518168041 (n=48)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
