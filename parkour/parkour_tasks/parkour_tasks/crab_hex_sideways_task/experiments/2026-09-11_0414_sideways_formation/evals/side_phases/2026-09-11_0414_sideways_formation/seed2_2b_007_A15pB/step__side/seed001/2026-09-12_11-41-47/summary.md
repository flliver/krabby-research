```
=== crab-hex gait eval ===
scenario   : step__side  (Isaac-Crab-Hex-Side-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_side_teacher/2026-09-12_05-24-07/model_14997.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.3261164050892925  p25=0.2557251316859752  p75=0.3991732857594123
tippy_tap_fraction median=0.3716646989374262
slip_ratio         median=0.5076251348919538
tracking_ratio     median=0.5785982455419356  (achieved/commanded vx, walking holds; n=99)
schedule_completion_rate=0.84
terminations={'schedule_complete': 84, 'fall': 16}

by hold:
     stand: cmd 0.00 -> achieved 0.003 m/s | tripod median=0.0 (n=89)
     creep: cmd 0.25 -> achieved 0.196 m/s | tripod median=0.5668552250545446 (n=99)
       low: cmd 0.35 -> achieved 0.142 m/s | tripod median=0.4078011827987549 (n=92)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
