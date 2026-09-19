```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-27_20-06-16/model_4998.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.2560674512320852  p25=0.167313597821315  p75=0.3361815449834502
tippy_tap_fraction median=0.3931809376210771
slip_ratio         median=0.4704576450960737
tracking_ratio     median=0.4800950281937954  (achieved/commanded vx, walking holds; n=95)
schedule_completion_rate=0.41
terminations={'schedule_complete': 41, 'fall': 59}

by hold:
     stand: cmd 0.00 -> achieved 0.011 m/s | tripod median=0.10098074411086559 (n=100)
     creep: cmd 0.25 -> achieved 0.138 m/s | tripod median=0.3709800524560201 (n=93)
       low: cmd 0.35 -> achieved 0.134 m/s | tripod median=0.4561422134202811 (n=48)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
