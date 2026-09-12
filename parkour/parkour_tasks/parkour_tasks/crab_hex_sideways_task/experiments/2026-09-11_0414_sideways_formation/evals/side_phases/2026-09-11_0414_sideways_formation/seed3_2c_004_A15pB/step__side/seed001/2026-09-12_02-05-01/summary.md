```
=== crab-hex gait eval ===
scenario   : step__side  (Isaac-Crab-Hex-Side-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_side_teacher/2026-09-11_19-43-19/model_19996.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.2833950927863088  p25=0.19888431825333136  p75=0.3570876525260769
tippy_tap_fraction median=0.4
slip_ratio         median=0.4302082742415505
tracking_ratio     median=0.5488989560311999  (achieved/commanded vx, walking holds; n=96)
schedule_completion_rate=0.63
terminations={'schedule_complete': 63, 'fall': 37}

by hold:
     stand: cmd 0.00 -> achieved 0.006 m/s | tripod median=0.0 (n=96)
     creep: cmd 0.25 -> achieved 0.185 m/s | tripod median=0.5309249708387995 (n=96)
       low: cmd 0.35 -> achieved 0.147 m/s | tripod median=0.3340860242004069 (n=78)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
