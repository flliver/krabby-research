```
=== crab-hex gait eval ===
scenario   : slow__base  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-01_15-10-31/model_19996.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5556760862710926  p25=0.5061284542846516  p75=0.590379229128636
tippy_tap_fraction median=0.25068306010928965
slip_ratio         median=0.21087989133865825
tracking_ratio     median=0.4313321819420712  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}

by hold:
     stand: cmd 0.00 -> achieved 0.054 m/s | tripod median=0.5761608105876386 (n=100)
     creep: cmd 0.25 -> achieved 0.110 m/s | tripod median=0.5766752952902584 (n=100)
       low: cmd 0.35 -> achieved 0.147 m/s | tripod median=0.5406767286416625 (n=100)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
