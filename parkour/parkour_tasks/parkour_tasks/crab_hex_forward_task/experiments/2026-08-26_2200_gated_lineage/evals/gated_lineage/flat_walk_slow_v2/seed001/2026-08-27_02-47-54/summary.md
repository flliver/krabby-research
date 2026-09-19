```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-26_21-41-53/model_26899.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.5955016213271105  p25=0.5524299324654377  p75=0.6269580120130316
tippy_tap_fraction median=0.211566938143636
slip_ratio         median=0.2173407322118152
tracking_ratio     median=0.44085201956773135  (achieved/commanded vx, walking holds; n=99)
schedule_completion_rate=0.88
terminations={'schedule_complete': 88, 'fall': 12}

by hold:
     stand: cmd 0.00 -> achieved 0.028 m/s | tripod median=0.5802233925325024 (n=100)
     creep: cmd 0.25 -> achieved 0.119 m/s | tripod median=0.5912135134905527 (n=99)
       low: cmd 0.35 -> achieved 0.143 m/s | tripod median=0.6276676677381365 (n=89)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
