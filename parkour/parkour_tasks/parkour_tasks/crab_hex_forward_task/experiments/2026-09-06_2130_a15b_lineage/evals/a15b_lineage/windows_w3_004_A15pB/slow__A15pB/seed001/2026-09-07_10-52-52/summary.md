```
=== crab-hex gait eval ===
scenario   : slow__A15pB  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-07_04-38-50/model_19996.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.4996107192805027  p25=0.43463711079109985  p75=0.5361980378536697
tippy_tap_fraction median=0.2531254883575559
slip_ratio         median=0.1897797920074168
tracking_ratio     median=0.6486039625831577  (achieved/commanded vx, walking holds; n=96)
schedule_completion_rate=0.81
terminations={'schedule_complete': 81, 'fall': 19}

by hold:
     stand: cmd 0.00 -> achieved 0.129 m/s | tripod median=0.49681593601153284 (n=100)
     creep: cmd 0.25 -> achieved 0.186 m/s | tripod median=0.5134265480701097 (n=96)
       low: cmd 0.35 -> achieved 0.192 m/s | tripod median=0.47970221718708395 (n=82)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
