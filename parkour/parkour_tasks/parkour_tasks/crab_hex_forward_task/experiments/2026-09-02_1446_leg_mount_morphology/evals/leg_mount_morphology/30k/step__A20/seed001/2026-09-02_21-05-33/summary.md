```
=== crab-hex gait eval ===
scenario   : step__A20  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-02_00-42-53/model_29994.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.611736183277052  p25=0.566550135142102  p75=0.644746141564312
tippy_tap_fraction median=0.25602497096399535
slip_ratio         median=0.2718562635782123
tracking_ratio     median=0.3677853610198459  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.93
terminations={'schedule_complete': 93, 'fall': 7}

by hold:
     stand: cmd 0.00 -> achieved 0.065 m/s | tripod median=0.7139950343693338 (n=100)
     creep: cmd 0.25 -> achieved 0.105 m/s | tripod median=0.5744490982219023 (n=100)
       low: cmd 0.35 -> achieved 0.110 m/s | tripod median=0.5621403985561486 (n=96)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
