```
=== crab-hex gait eval ===
scenario   : step__A15pB  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-31_10-46-19/model_19996.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.632847060317872  p25=0.5862497449477322  p75=0.6818081631893147
tippy_tap_fraction median=0.24404424276800907
slip_ratio         median=0.21215093317983208
tracking_ratio     median=0.3831744749709822  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=1.0
terminations={'schedule_complete': 100}

by hold:
     stand: cmd 0.00 -> achieved 0.067 m/s | tripod median=0.6784004653495889 (n=100)
     creep: cmd 0.25 -> achieved 0.101 m/s | tripod median=0.6418561521128117 (n=100)
       low: cmd 0.35 -> achieved 0.126 m/s | tripod median=0.6196109539122393 (n=100)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
