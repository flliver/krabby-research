```
=== crab-hex gait eval ===
scenario   : slow__A20pB  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-02_00-42-53/model_29994.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6597082568227517  p25=0.6428074899111784  p75=0.6742858478233443
tippy_tap_fraction median=0.24592918066683897
slip_ratio         median=0.237921696706095
tracking_ratio     median=0.3737868478050371  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}

by hold:
     stand: cmd 0.00 -> achieved 0.059 m/s | tripod median=0.7192999124037651 (n=100)
     creep: cmd 0.25 -> achieved 0.101 m/s | tripod median=0.6302246039631108 (n=100)
       low: cmd 0.35 -> achieved 0.121 m/s | tripod median=0.6398132038810196 (n=100)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
