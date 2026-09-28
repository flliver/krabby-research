```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-03_23-07-36/model_24995.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5686751202121763  p25=0.5387515956952527  p75=0.5858255239527956
tippy_tap_fraction median=0.23353293413173654
slip_ratio         median=0.21849529175149585
tracking_ratio     median=0.44887353784847894  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.9
terminations={'schedule_complete': 90, 'fall': 10}

by hold:
     stand: cmd 0.00 -> achieved 0.062 m/s | tripod median=0.6958754429003948 (n=100)
     creep: cmd 0.25 -> achieved 0.118 m/s | tripod median=0.505559686998954 (n=100)
       low: cmd 0.35 -> achieved 0.151 m/s | tripod median=0.4907069967184788 (n=91)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
