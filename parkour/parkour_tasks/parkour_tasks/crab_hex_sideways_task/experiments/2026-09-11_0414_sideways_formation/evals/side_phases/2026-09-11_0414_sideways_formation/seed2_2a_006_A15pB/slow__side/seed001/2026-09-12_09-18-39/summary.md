```
=== crab-hex gait eval ===
scenario   : slow__side  (Isaac-Crab-Hex-Side-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_side_teacher/2026-09-12_03-04-37/model_9998.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.4532945572694276  p25=0.4131325204309677  p75=0.48777235291256976
tippy_tap_fraction median=0.3780780780780781
slip_ratio         median=0.4181115558128986
tracking_ratio     median=0.7402557648331153  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.97
terminations={'schedule_complete': 97, 'fall': 3}

by hold:
     stand: cmd 0.00 -> achieved 0.006 m/s | tripod median=0.0 (n=94)
     creep: cmd 0.25 -> achieved 0.219 m/s | tripod median=0.6436766589099451 (n=100)
       low: cmd 0.35 -> achieved 0.219 m/s | tripod median=0.7004816244133649 (n=97)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
