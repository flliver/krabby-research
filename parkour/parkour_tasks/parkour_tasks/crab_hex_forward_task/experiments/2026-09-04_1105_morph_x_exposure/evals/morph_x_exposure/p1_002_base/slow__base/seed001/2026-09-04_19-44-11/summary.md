```
=== crab-hex gait eval ===
scenario   : slow__base  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-04_13-18-07/model_4999.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.44398876607438  p25=0.36077097658587043  p75=0.535647398154717
tippy_tap_fraction median=0.2644284128745838
slip_ratio         median=0.26479966064888916
tracking_ratio     median=0.5962714947815236  (achieved/commanded vx, walking holds; n=95)
schedule_completion_rate=0.46
terminations={'fall': 54, 'schedule_complete': 46}

by hold:
     stand: cmd 0.00 -> achieved 0.062 m/s | tripod median=0.38140747602235364 (n=100)
     creep: cmd 0.25 -> achieved 0.163 m/s | tripod median=0.49277803633361017 (n=94)
       low: cmd 0.35 -> achieved 0.168 m/s | tripod median=0.6145647221651813 (n=58)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
