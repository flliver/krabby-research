```
=== crab-hex gait eval ===
scenario   : slow__side  (Isaac-Crab-Hex-Side-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_side_flat_walk/2026-09-11_10-57-55/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.3553051200820655  p25=0.318652414482242  p75=0.3856467171002697
tippy_tap_fraction median=0.4141743247006405
slip_ratio         median=0.4326199344035278
tracking_ratio     median=0.6338797944379563  (achieved/commanded vx, walking holds; n=96)
schedule_completion_rate=0.93
terminations={'schedule_complete': 93, 'fall': 7}

by hold:
     stand: cmd 0.00 -> achieved 0.006 m/s | tripod median=0.00096882961832296 (n=99)
     creep: cmd 0.25 -> achieved 0.184 m/s | tripod median=0.5374012644060795 (n=96)
       low: cmd 0.35 -> achieved 0.189 m/s | tripod median=0.5326433683540394 (n=93)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
