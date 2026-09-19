```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Student-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_student/2026-09-09_02-06-51/model_29994.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.4523305779521123  p25=0.3968427417153732  p75=0.5097742013855135
tippy_tap_fraction median=0.2434164934164934
slip_ratio         median=0.19240272870702735
tracking_ratio     median=0.5562445043752283  (achieved/commanded vx, walking holds; n=91)
schedule_completion_rate=0.52
terminations={'schedule_complete': 52, 'fall': 48}

by hold:
     stand: cmd 0.00 -> achieved 0.110 m/s | tripod median=0.4491292518014564 (n=100)
     creep: cmd 0.25 -> achieved 0.155 m/s | tripod median=0.47462647958919474 (n=90)
       low: cmd 0.35 -> achieved 0.169 m/s | tripod median=0.4701234822499196 (n=62)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
