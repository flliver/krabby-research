```
=== crab-hex gait eval ===
scenario   : step__side  (Isaac-Crab-Hex-Side-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_side_flat_walk/2026-09-12_00-31-09/model_4999.pt
episodes   : 100  unscored: 5
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.28370940729230115  p25=0.20125440446681048  p75=0.36601990206618296
tippy_tap_fraction median=0.4014566148063068
slip_ratio         median=0.4586595943216506
tracking_ratio     median=0.5308019013095473  (achieved/commanded vx, walking holds; n=85)
schedule_completion_rate=0.6
terminations={'schedule_complete': 60, 'fall': 40}

by hold:
     stand: cmd 0.00 -> achieved 0.001 m/s | tripod median=0.0 (n=89)
     creep: cmd 0.25 -> achieved 0.170 m/s | tripod median=0.5039485169395372 (n=85)
       low: cmd 0.35 -> achieved 0.129 m/s | tripod median=0.4433014046723247 (n=70)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
