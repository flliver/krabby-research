```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-04_22-02-27/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.4557939410845019  p25=0.4088023402404712  p75=0.5264251296398268
tippy_tap_fraction median=0.23085635769459298
slip_ratio         median=0.2033049245312888
tracking_ratio     median=0.6058429478928884  (achieved/commanded vx, walking holds; n=95)
schedule_completion_rate=0.49
terminations={'schedule_complete': 49, 'fall': 51}

by hold:
     stand: cmd 0.00 -> achieved 0.043 m/s | tripod median=0.44379904035896645 (n=100)
     creep: cmd 0.25 -> achieved 0.181 m/s | tripod median=0.5724944728015304 (n=95)
       low: cmd 0.35 -> achieved 0.170 m/s | tripod median=0.35457847462847114 (n=75)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
