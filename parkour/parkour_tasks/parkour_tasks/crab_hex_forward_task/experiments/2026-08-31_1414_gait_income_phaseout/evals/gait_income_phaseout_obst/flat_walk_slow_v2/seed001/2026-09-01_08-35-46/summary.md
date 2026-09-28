```
=== crab-hex gait eval ===
scenario   : flat_walk_slow_v2  (Isaac-Crab-Hex-Flat-Walk-v0)
checkpoint : /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-01_02-22-26/model_9998.pt
episodes   : 100  unscored: 0
terrain    : raycast_mesh
cmd override max deviation: 0.000e+00

tripod_score      median=0.538668229973225  p25=0.4721591277997402  p75=0.5795680260099245
tippy_tap_fraction median=0.24241978757028926
slip_ratio         median=0.2295593241537015
tracking_ratio     median=0.4346627437012953  (achieved/commanded vx, walking holds; n=100)
schedule_completion_rate=0.73
terminations={'schedule_complete': 73, 'fall': 27}

by hold:
     stand: cmd 0.00 -> achieved 0.037 m/s | tripod median=0.46637310661869014 (n=100)
     creep: cmd 0.25 -> achieved 0.115 m/s | tripod median=0.6134190110071851 (n=100)
       low: cmd 0.35 -> achieved 0.140 m/s | tripod median=0.5246240828153201 (n=82)

[WARN] num_prop=75 but observation width is 1149; train and play share this slicing so it is self-consistent, but the 'scan' slice is not the height scan.
```
