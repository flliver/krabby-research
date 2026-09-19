```
=== crab-hex gait eval ===
scenario   : slow__A20  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-31_10-46-19/model_19996.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.6636535011140339  p25=0.6385769845800636  p75=0.6875115631890638
tippy_tap_fraction median=0.2346441947565543
slip_ratio         median=0.2055523436794943
tracking_ratio     median=0.4087363431288999  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}

by hold:
     stand: cmd 0.00 -> achieved 0.055 m/s | tripod median=0.6809051767698365 (n=100)
     creep: cmd 0.25 -> achieved 0.097 m/s | tripod median=0.6446335869281982 (n=100)
       low: cmd 0.35 -> achieved 0.150 m/s | tripod median=0.6671072690617736 (n=100)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
