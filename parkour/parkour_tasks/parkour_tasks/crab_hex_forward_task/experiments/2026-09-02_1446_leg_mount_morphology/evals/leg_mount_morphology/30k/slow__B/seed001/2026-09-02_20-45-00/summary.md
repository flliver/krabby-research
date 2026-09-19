```
=== crab-hex gait eval ===
scenario   : slow__B  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-02_00-42-53/model_29994.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6202708555951815  p25=0.6030315308846913  p75=0.6504614698438618
tippy_tap_fraction median=0.24443638122883404
slip_ratio         median=0.23001657809520112
tracking_ratio     median=0.38866248185378593  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.98
terminations={'schedule_complete': 98, 'fall': 2}

by hold:
     stand: cmd 0.00 -> achieved 0.064 m/s | tripod median=0.734800902929852 (n=100)
     creep: cmd 0.25 -> achieved 0.107 m/s | tripod median=0.5872893049199732 (n=100)
       low: cmd 0.35 -> achieved 0.123 m/s | tripod median=0.5604982944391438 (n=98)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
