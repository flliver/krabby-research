```
=== crab-hex gait eval ===
scenario   : slow__side  (Isaac-Crab-Hex-Side-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_side_teacher/2026-09-12_03-04-37/model_9800.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.4662531714598833  p25=0.43697935709173125  p75=0.5017454683424036
tippy_tap_fraction median=0.386309612241866
slip_ratio         median=0.44864903152612023
tracking_ratio     median=0.7413011464823199  (achieved/commanded vx, walking holds; n=99)
schedule_completion_rate=0.98
terminations={'schedule_complete': 98, 'fall': 2}

by hold:
     stand: cmd 0.00 -> achieved 0.004 m/s | tripod median=0.0 (n=89)
     creep: cmd 0.25 -> achieved 0.215 m/s | tripod median=0.6892019335422312 (n=99)
       low: cmd 0.35 -> achieved 0.221 m/s | tripod median=0.6943268243562791 (n=98)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
