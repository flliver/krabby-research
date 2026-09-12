```
=== crab-hex gait eval ===
scenario   : step__side  (Isaac-Crab-Hex-Side-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_side_teacher/2026-09-11_14-50-01/model_9998.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.29014417021935845  p25=0.23571322125337352  p75=0.3539759722100077
tippy_tap_fraction median=0.3954356126345019
slip_ratio         median=0.44884736740507125
tracking_ratio     median=0.535669260627089  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.82
terminations={'schedule_complete': 82, 'fall': 18}

by hold:
     stand: cmd 0.00 -> achieved 0.002 m/s | tripod median=0.0 (n=88)
     creep: cmd 0.25 -> achieved 0.176 m/s | tripod median=0.5329924153232646 (n=100)
       low: cmd 0.35 -> achieved 0.133 m/s | tripod median=0.3534937145677488 (n=91)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
