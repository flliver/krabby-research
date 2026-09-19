```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-03_23-07-36/model_24995.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5817325498964465  p25=0.5342845697916082  p75=0.6310203434336832
tippy_tap_fraction median=0.2334890965732087
slip_ratio         median=0.21130394393979762
tracking_ratio     median=0.5380560423938294  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.33
terminations={'schedule_complete': 33, 'fall': 67}

by hold:
     stand: cmd 0.00 -> achieved 0.072 m/s | tripod median=0.6892960901295426 (n=100)
     creep: cmd 0.25 -> achieved 0.135 m/s | tripod median=0.4869721897592062 (n=100)
       low: cmd 0.35 -> achieved 0.159 m/s | tripod median=0.4174299206126204 (n=55)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
