```
=== crab-hex gait eval ===
scenario   : fwd__A10  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-02_00-42-53/model_29994.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.47996449603132174  p25=0.4623457371683196  p75=0.5102698026411417
tippy_tap_fraction median=0.3056568303520565
slip_ratio         median=0.2672262229838226
tracking_ratio     median=0.2811982808953746  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.67
terminations={'schedule_complete': 67, 'fall': 33}

by hold:
       low: cmd 0.30 -> achieved 0.112 m/s | tripod median=0.5677719975040376 (n=100)
       mid: cmd 0.47 -> achieved 0.121 m/s | tripod median=0.46859015955797345 (n=100)
      high: cmd 0.65 -> achieved 0.133 m/s | tripod median=0.39923931936935525 (n=97)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
