```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-01_09-23-40/model_14996.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5969103486552686  p25=0.5651196578923724  p75=0.6244632326855285
tippy_tap_fraction median=0.22946766840043664
slip_ratio         median=0.19147485508645762
tracking_ratio     median=0.45473507210013603  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.99
terminations={'schedule_complete': 99, 'fall': 1}

by hold:
     stand: cmd 0.00 -> achieved 0.056 m/s | tripod median=0.575692919637834 (n=100)
     creep: cmd 0.25 -> achieved 0.117 m/s | tripod median=0.6386574081516755 (n=100)
       low: cmd 0.35 -> achieved 0.154 m/s | tripod median=0.582914906397324 (n=99)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
