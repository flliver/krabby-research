```
=== crab-hex gait eval ===
scenario   : slow__A15pB  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-06_16-03-30/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.47642717083281494  p25=0.36499835564358796  p75=0.5453571904882817
tippy_tap_fraction median=0.2696178937558248
slip_ratio         median=0.26310102695702303
tracking_ratio     median=0.7132862684459824  (achieved/commanded vx, walking holds; n=77)
schedule_completion_rate=0.58
terminations={'fall': 42, 'schedule_complete': 58}

by hold:
     stand: cmd 0.00 -> achieved 0.029 m/s | tripod median=0.3746269737397098 (n=100)
     creep: cmd 0.25 -> achieved 0.197 m/s | tripod median=0.5251532238271323 (n=77)
       low: cmd 0.35 -> achieved 0.212 m/s | tripod median=0.6437386082129481 (n=62)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
