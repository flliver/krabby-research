```
=== crab-hex gait eval ===
scenario   : slow__A20  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-05_00-31-35/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.466171302788711  p25=0.387626621440217  p75=0.5256893082065045
tippy_tap_fraction median=0.2517244660101803
slip_ratio         median=0.22793403381913901
tracking_ratio     median=0.6167020986379977  (achieved/commanded vx, walking holds; n=70)
schedule_completion_rate=0.5
terminations={'fall': 50, 'schedule_complete': 50}

by hold:
     stand: cmd 0.00 -> achieved 0.030 m/s | tripod median=0.3545025014349031 (n=100)
     creep: cmd 0.25 -> achieved 0.178 m/s | tripod median=0.5901117277177348 (n=69)
       low: cmd 0.35 -> achieved 0.176 m/s | tripod median=0.5398617169854194 (n=63)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
